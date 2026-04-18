"""
Part 3: PageRank on Spark.
Implements the iterative PageRank with β = 0.8 for 40 iterations.

PageRank formula:
    r_0     = (1/n) * A           (A = n x 1 all-ones vector)
    r_{i+1} = ((1 - β) / n) * A + β * M * r_i
Where M_{i,j} = 1 / deg(i) for (i -> j) in E, else 0.
Duplicate directed edges are collapsed via .distinct().
Dangling-node leaked mass is redistributed uniformly to keep sum(r) == 1.

Usage:
    python3 part3_pagerank.py
    (expects graphs/small.txt and graphs/whole.txt next to this script,
     or override with GRAPHS_DIR env var)
"""
import os
import sys

from pyspark import SparkContext, SparkConf


def build_graph(sc, path):
    raw = sc.textFile(path)
    edges = (raw.map(lambda s: s.strip())
                .filter(lambda s: s and not s.startswith("#"))
                .map(lambda s: s.split())
                .filter(lambda p: len(p) >= 2)
                .map(lambda p: (int(p[0]), int(p[1])))
                .distinct())
    nodes = edges.flatMap(lambda e: [e[0], e[1]]).distinct().collect()
    n = len(nodes)

    # (src, [dsts]); ensure every node has an entry (sinks get [])
    links = edges.groupByKey().mapValues(lambda it: list(set(it)))
    all_nodes_pair = sc.parallelize([(x, []) for x in nodes])
    links = (links.union(all_nodes_pair)
                  .reduceByKey(lambda a, b: list(set(a) | set(b)))
                  .cache())
    return links, nodes, n


def _contrib(kv, beta):
    node, (neighbours, r) = kv
    if not neighbours:
        return []  # dangling handled separately
    share = beta * r / len(neighbours)
    return [(nb, share) for nb in neighbours]


def pagerank(sc, edges_path, beta=0.8, iters=40, verbose=True):
    links, nodes, n = build_graph(sc, edges_path)
    uniq_edges = links.map(lambda kv: len(kv[1])).sum()
    if verbose:
        print(f"[GRAPH] n = {n} nodes, unique edges = {uniq_edges}")

    ranks = links.mapValues(lambda _: 1.0 / n).cache()
    teleport = (1.0 - beta) / n

    for it in range(1, iters + 1):
        contribs = links.join(ranks).flatMap(lambda kv: _contrib(kv, beta))
        summed = contribs.reduceByKey(lambda a, b: a + b)

        dangling_mass = (links.filter(lambda kv: len(kv[1]) == 0)
                             .map(lambda kv: (kv[0], None))
                             .leftOuterJoin(ranks)
                             .map(lambda kv: kv[1][1] or 0.0)
                             .sum())
        leaked = beta * dangling_mass / n

        all_nodes = links.mapValues(lambda _: 0.0)
        ranks = (all_nodes.leftOuterJoin(summed)
                          .mapValues(lambda v: (v[1] or 0.0) + teleport + leaked)
                          .cache())

        if verbose and it in (1, 5, 10, 20, 40, iters):
            top = ranks.takeOrdered(1, key=lambda kv: -kv[1])[0]
            total = ranks.map(lambda kv: kv[1]).sum()
            print(f"[iter {it:2d}] top = (node {top[0]}, r = {top[1]:.6f}), "
                  f"sum(r) = {total:.6f}")

    return ranks


def report_top_bottom(ranks, k=5):
    top_k = ranks.takeOrdered(k, key=lambda kv: -kv[1])
    bot_k = ranks.takeOrdered(k, key=lambda kv: kv[1])
    print(f"Top {k} nodes (highest PageRank):")
    for node, r in top_k:
        print(f"  node {node:>5} -> {r:.6f}")
    print(f"Bottom {k} nodes (lowest PageRank):")
    for node, r in bot_k:
        print(f"  node {node:>5} -> {r:.6f}")
    return top_k, bot_k


if __name__ == "__main__":
    BASE = os.environ.get(
        "GRAPHS_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs"),
    )

    conf = (SparkConf().setAppName("Part3-PageRank")
            .setMaster("local[*]")
            .set("spark.driver.bindAddress", "127.0.0.1")
            .set("spark.ui.showConsoleProgress", "false"))
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("ERROR")

    for which in ("small.txt", "whole.txt"):
        path = os.path.join(BASE, which)
        print("=" * 60)
        print(f"Dataset: {which}")
        ranks = pagerank(sc, path, beta=0.8, iters=40)
        report_top_bottom(ranks, k=5)

    sc.stop()
