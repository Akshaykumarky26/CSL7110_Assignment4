"""
Part 1: Clustering on Spark
Implementation of Farthest-First Traversal (k-center) and K-Means++ algorithms.

Complexity note:
  * kcenter(P, k)  runs in O(|P| * k)  -- k rounds x one scan of |P| each.
  * kmeansPP(P, k) runs in O(|P| * k)  -- k rounds x one scan of |P| each.

Usage:
    python3 part1_clustering.py <k> <k1>
    (k < k1; defaults to k=10 k1=50)

The script expects spambase.data to live next to it, i.e. the same directory
as this script, unless the SPAMBASE_PATH env var is set.
"""
import os
import sys
import time
import math
import random

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vector, Vectors


# -------------------------------------------------------------------- I/O
def readVectorsSeq(filename):
    """Read points from file; one point per line, comma-separated values.
    Returns a list of Vector.

    The original Spambase file has 57 feature attributes + 1 {0,1} class
    column. The assignment states the data has 58 dimensions, so we keep
    all 58 numeric columns as coordinates.
    """
    vectors = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feats = [float(x) for x in line.split(",")]
            vectors.append(Vectors.dense(feats))
    return vectors


def _sq_dist(a, b):
    """Squared L2 distance between two dense vectors.
    Uses Vectors.sqdist when available; falls back to numpy otherwise so the
    helper works across PySpark versions.
    """
    try:
        return float(Vectors.sqdist(a, b))
    except AttributeError:
        aa = a.toArray() if hasattr(a, "toArray") else a
        bb = b.toArray() if hasattr(b, "toArray") else b
        diff = aa - bb
        return float((diff * diff).sum())


# ------------------------------------------------------------ Algorithms
def kcenter(P, k):
    """Farthest-First Traversal algorithm. O(|P| * k).
    Returns k centers selected from P.
    """
    n = len(P)
    if k <= 0 or n == 0:
        return []
    centers = [P[0]]  # arbitrary first point for reproducibility
    min_sq = [_sq_dist(P[i], centers[0]) for i in range(n)]
    for _ in range(1, k):
        farthest_idx, farthest_val = 0, -1.0
        for i in range(n):  # one scan of |P|
            if min_sq[i] > farthest_val:
                farthest_val, farthest_idx = min_sq[i], i
        c = P[farthest_idx]
        centers.append(c)
        for i in range(n):  # update nearest-center distance
            d = _sq_dist(P[i], c)
            if d < min_sq[i]:
                min_sq[i] = d
    return centers


def kmeansPP(P, k):
    """K-Means++ seeding. O(|P| * k).
    Returns k centers sampled with D^2 weighting.
    """
    n = len(P)
    if k <= 0 or n == 0:
        return []
    random.seed(42)
    first = random.randint(0, n - 1)
    centers = [P[first]]
    min_sq = [_sq_dist(P[i], centers[0]) for i in range(n)]
    for _ in range(1, k):
        total = sum(min_sq)
        if total <= 0.0:
            idx = random.randint(0, n - 1)
        else:
            r = random.random() * total
            cum, idx = 0.0, n - 1
            for i in range(n):
                cum += min_sq[i]
                if cum >= r:
                    idx = i
                    break
        c = P[idx]
        centers.append(c)
        for i in range(n):
            d = _sq_dist(P[i], c)
            if d < min_sq[i]:
                min_sq[i] = d
    return centers


def kmeansObj(P, C):
    """Average squared distance of a point of P from its nearest center."""
    if not P:
        return 0.0
    total = 0.0
    for p in P:
        best = float("inf")
        for c in C:
            d = _sq_dist(p, c)
            if d < best:
                best = d
        total += best
    return total / len(P)


# ------------------------------------------------------------- Driver
def main(path, k, k1):
    P = readVectorsSeq(path)
    print(f"[INFO] Loaded |P| = {len(P)} points with dimension {len(P[0])}.")

    # 1. kcenter(P, k)
    t0 = time.time()
    C_kc = kcenter(P, k)
    t1 = time.time()
    print(f"[kcenter] k = {k}: running time = {t1 - t0:.4f} s, centers = {len(C_kc)}")

    # 2. kmeansPP(P, k) -> kmeansObj(P, C)
    t0 = time.time()
    C_pp = kmeansPP(P, k)
    t1 = time.time()
    obj_pp = kmeansObj(P, C_pp)
    print(f"[kmeansPP] k = {k}: time = {t1 - t0:.4f} s, kmeansObj = {obj_pp:.6f}")

    # 3. Coreset: kcenter(P, k1) -> X, kmeansPP(X, k) -> C, kmeansObj(P, C)
    t0 = time.time()
    X = kcenter(P, k1)
    t1 = time.time()
    print(f"[kcenter] k1 = {k1}: time = {t1 - t0:.4f} s, |X| = {len(X)}")
    C_cs = kmeansPP(X, k)
    obj_cs = kmeansObj(P, C_cs)
    print(f"[coreset] kmeansPP(X, {k}) -> kmeansObj(P, C) = {obj_cs:.6f}")

    return {"obj_pp": obj_pp, "obj_coreset": obj_cs}


if __name__ == "__main__":
    # Portable default: look for spambase.data next to this script.
    BASE = os.path.dirname(os.path.abspath(__file__))
    path = os.environ.get(
        "SPAMBASE_PATH",
        os.path.join(BASE, "spambase.data"),
    )
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    K1 = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    assert K < K1, "k must be < k1"

    conf = (SparkConf()
            .setAppName("Part1-Clustering")
            .setMaster("local[*]")
            .set("spark.driver.bindAddress", "127.0.0.1")
            .set("spark.ui.showConsoleProgress", "false"))
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("ERROR")

    main(path, K, K1)

    sc.stop()
