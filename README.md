# CSL7110 Assignment 4 — Clustering, Web Search and PageRank

**Course:** Machine Learning with Big Data  
**Assignment:** Assignment 4  
**Submitted by:** Akshay Kumar  
**Roll Number:** M25DE1028  

## Overview

This repository contains the complete implementation and report for **CSL7110 Assignment 4: Clustering, Web Search and PageRank**.

The assignment is divided into three major parts:

1. **Clustering on Spambase data**
   - Farthest-First Traversal / k-center
   - k-Means++ seeding
   - k-means objective computation
   - k-center based coreset experiment

2. **Web Search using Inverted Index**
   - Custom implementation of required classes
   - Page indexing
   - Word-position tracking
   - Query handling using `actions.txt`
   - Output validation against `answers.txt`

3. **PageRank using PySpark**
   - Iterative PageRank computation
   - Duplicate edge removal
   - Dangling-node handling
   - Top-5 and bottom-5 PageRank reporting

## Repository Structure

```text
CSL7110_Assignment4/
├── CSL7110_Assignment4.ipynb
├── M25DE1028_CSL7110_Assignment4.pdf
├── M25DE1028_CSL7110_Assignment4.docx
├── part1_clustering.py
├── part2_websearch.py
├── part3_pagerank.py
├── spambase.data
├── actions.txt
├── answers.txt
├── graphs/
│   ├── small.txt
│   └── whole.txt
├── webpages/
│   ├── references
│   ├── stack_cprogramming
│   ├── stack_datastructure_wiki
│   ├── stack_oracle
│   ├── stackoverflow
│   ├── stacklighting
│   └── stackmagazine
└── README.md
```

## Environment

The implementation was tested with:

```text
Python 3.10.12
PySpark 3.5.3
OpenJDK 11
NumPy
```

PySpark is used in local mode:

```text
master = local[*]
spark.driver.bindAddress = 127.0.0.1
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install pyspark==3.5.3 numpy jupyter ipykernel
```

If using Jupyter Notebook, register the virtual environment as a kernel:

```bash
python -m ipykernel install --user --name assignment4-venv --display-name "Python (Assignment 4)"
```

## Java Requirement for Spark

PySpark requires Java to be installed.

Check Java using:

```bash
java -version
```

If Java is not installed, install a JDK such as OpenJDK / Temurin JDK and ensure `JAVA_HOME` is configured.

Example:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
export PATH="$JAVA_HOME/bin:$PATH"
```

# Part 1 — Clustering

## Objective

Part 1 implements two clustering primitives:

1. `kcenter(P, k)`
2. `kmeansPP(P, k)`

The program also computes the k-means objective value using:

```python
kmeansObj(P, C)
```

where `P` is the set of points and `C` is the set of selected centers.

## Dataset

The file used is:

```text
spambase.data
```

The assignment specifies the dataset as having **4601 points in 58 dimensions**.

The original Spambase dataset contains 57 continuous attributes and one final class-label column. Since the assignment explicitly states 58 dimensions, all 58 numeric columns are treated as coordinates.

## Implemented Functions

```python
readVectorsSeq(filename)
```

Reads the dataset and returns a list of PySpark dense vectors.

```python
kcenter(P, k)
```

Implements Farthest-First Traversal.

Time complexity:

```text
O(|P| · k)
```

```python
kmeansPP(P, k)
```

Implements k-Means++ seeding using D² sampling.

Time complexity:

```text
O(|P| · k)
```

```python
kmeansObj(P, C)
```

Computes the average squared distance of each point from its nearest center.

## How to Run Part 1

```bash
python3 part1_clustering.py 10 50
python3 part1_clustering.py 15 100
python3 part1_clustering.py 5 200
```

The command-line arguments are:

```text
k  = number of final centers
k1 = number of k-center coreset points
```

with:

```text
k < k1
```

## Part 1 Experiments

Three experiments were performed:

```text
Experiment 1: k = 10, k1 = 50
Experiment 2: k = 15, k1 = 100
Experiment 3: k = 5,  k1 = 200
```

Each experiment reports:

1. Running time of `kcenter(P, k)`
2. k-means objective after direct `kmeansPP(P, k)`
3. k-means objective after `kcenter(P, k1)` followed by `kmeansPP(X, k)`

## Part 1 Result Summary

```text
Experiment 1: k = 10, k1 = 50
[kcenter ] k = 10: time = 0.0651 s, centers = 10
[kmeansPP] k = 10: time = 0.0647 s, kmeansObj(direct)  = 31251.6036
[coreset ] k1 = 50: kcenter time = 0.3312 s, |X| = 50
           kmeansPP(X, 10) -> kmeansObj(coreset) = 99261.9237

Experiment 2: k = 15, k1 = 100
[kcenter ] k = 15: time = 0.1029 s, centers = 15
[kmeansPP] k = 15: time = 0.0998 s, kmeansObj(direct)  = 23079.7012
[coreset ] k1 = 100: kcenter time = 0.6621 s, |X| = 100
           kmeansPP(X, 15) -> kmeansObj(coreset) = 163616.0197

Experiment 3: k = 5, k1 = 200
[kcenter ] k =  5: time = 0.0328 s, centers = 5
[kmeansPP] k =  5: time = 0.0340 s, kmeansObj(direct)  = 77727.0389
[coreset ] k1 = 200: kcenter time = 1.3103 s, |X| = 200
           kmeansPP(X, 5) -> kmeansObj(coreset) = 157413.2084
```

## Part 1 Observations

- The running time of `kcenter` grows roughly linearly with `k`, matching the expected `O(|P| · k)` complexity.
- Direct k-Means++ gives lower objective values than the coreset-based approach in these experiments.
- Farthest-First Traversal tends to select extreme/outlier points, so the selected coreset may not always represent dense regions well.
- The coreset approach is distribution-sensitive and depends strongly on the choice of `k1`.

# Part 2 — Web Search Inverted Index

## Objective

Part 2 implements a simple web-search system using an inverted index.

The system reads page files from the `webpages/` directory and processes commands from:

```text
actions.txt
```

The output is compared with:

```text
answers.txt
```

## Implemented Classes

The following required classes are implemented:

```text
MySet
Position
WordEntry
PageIndex
PageEntry
MyHashTable
InvertedPageIndex
SearchEngine
```

## Supported Actions

The search engine supports:

```text
addPage pageName
```

Adds a webpage to the inverted index.

```text
queryFindPagesWhichContainWord word
```

Returns pages containing the given word.

```text
queryFindPositionsOfWordInAPage word pageName
```

Returns positions of a word within a given page.

## Text Processing Rules

The implementation follows the assignment rules:

1. All words are converted to lowercase.
2. Connector words are not stored.
3. Connector words are still counted for word positions.
4. Punctuation is replaced by spaces.
5. Selected plural/singular forms are normalised:
   - `stack` / `stacks`
   - `structure` / `structures`
   - `application` / `applications`

Connector words skipped from storage:

```text
a, an, the, they, these, this, for, is, are, was,
of, or, and, does, will, whose
```

## Webpage Files

The webpage corpus is stored inside:

```text
webpages/
```

It contains:

```text
references
stack_cprogramming
stack_datastructure_wiki
stack_oracle
stackoverflow
stacklighting
stackmagazine
```

Only the pages added through `actions.txt` are indexed during execution.

## How to Run Part 2

```bash
python3 part2_websearch.py
```

The script executes all actions from `actions.txt`, prints the results, and compares them with `answers.txt`.

Expected final message:

```text
ALL PASSED
```

## Part 2 Output Summary

The 11 query outputs are:

```text
No webpage contains word delhi
stack_datastructure_wiki
stack_datastructure_wiki
Webpage stack_datastructure_wiki does not contain word magazines
No webpage contains word allain
stack_cprogramming
stack_cprogramming
stack_cprogramming
stack_oracle
stack_cprogramming, stack_datastructure_wiki, stackoverflow
stackmagazine
```

All outputs match `answers.txt`.

## Part 2 Observations

- Connector words are skipped from storage but still counted for word positions.
- Word positions therefore remain consistent with the original page text.
- Plural/singular normalisation is necessary for matching words such as `stack/stacks`.
- The inverted index provides efficient word-to-page lookup.
- A `tfidfScore` helper is also included for ranked word scoring.

# Part 3 — PageRank on Spark

## Objective

Part 3 implements iterative PageRank using PySpark.

The graph is represented using an edge-list format:

```text
source_node destination_node
```

The program processes:

```text
graphs/small.txt
graphs/whole.txt
```

## PageRank Formula

The implementation uses:

```text
r₀ = (1/n) · A
```

and iteratively computes:

```text
rᵢ = ((1 - β) / n) · A + β · M · rᵢ₋₁
```

where:

```text
β = 0.8
n = number of nodes
M = transition matrix
A = all-ones vector
```

The algorithm runs for:

```text
40 iterations
```

## Important Implementation Details

The PageRank implementation:

1. Removes duplicate directed edges using `.distinct()`.
2. Builds adjacency lists as RDDs.
3. Handles dangling nodes with out-degree 0.
4. Redistributes dangling-node leaked mass uniformly.
5. Verifies that the PageRank vector sums to 1.0.
6. Reports top-5 and bottom-5 nodes.

## Graph Files

The graph files are stored inside:

```text
graphs/
```

They contain:

```text
small.txt
whole.txt
```

Each line represents a directed edge:

```text
source destination
```

Example:

```text
100 1
13 1
28 1
```

This represents:

```text
100 → 1
13 → 1
28 → 1
```

## How to Run Part 3

```bash
python3 part3_pagerank.py
```

This runs PageRank on both:

```text
small.txt
whole.txt
```

## Part 3 Results

### small.txt

```text
n = 100 nodes
unique edges = 950
top PageRank ≈ 0.0357
```

Iteration summary:

```text
[iter  1] top = (node 14, r = 0.035585), sum(r) = 1.000000
[iter  5] top = (node 53, r = 0.035722), sum(r) = 1.000000
[iter 10] top = (node 53, r = 0.035731), sum(r) = 1.000000
[iter 20] top = (node 53, r = 0.035731), sum(r) = 1.000000
[iter 40] top = (node 53, r = 0.035731), sum(r) = 1.000000
```

Top 5 nodes:

```text
53
14
40
1
27
```

Bottom 5 nodes:

```text
85
59
81
37
89
```

### whole.txt

```text
n = 1000 nodes
unique edges = 8161
```

Iteration summary:

```text
[iter  1] top = (node 263, r = 0.001876), sum(r) = 1.000000
[iter  5] top = (node 263, r = 0.002021), sum(r) = 1.000000
[iter 10] top = (node 263, r = 0.002020), sum(r) = 1.000000
[iter 20] top = (node 263, r = 0.002020), sum(r) = 1.000000
[iter 40] top = (node 263, r = 0.002020), sum(r) = 1.000000
```

Top 5 nodes:

```text
263
537
965
243
285
```

Bottom 5 nodes:

```text
558
93
62
424
408
```

## Part 3 Observations

- PageRank converges quickly on both graphs.
- The top node stabilises within a few iterations.
- The rank vector sum remains 1.0 due to explicit dangling-node mass redistribution.
- The transition matrix is never materialised as a dense matrix.
- The graph is processed as RDD-based adjacency lists, which is suitable for Spark.

## Notebook

The notebook:

```text
CSL7110_Assignment4.ipynb
```

contains the complete executed workflow for all three parts.

It includes:

1. Environment setup
2. Part 1 experiments
3. Part 2 output validation
4. Part 3 PageRank runs
5. Final outputs and observations

## Report

The final report is available as:

```text
M25DE1028_CSL7110_Assignment4.pdf
```

A Word version is also included:

```text
M25DE1028_CSL7110_Assignment4.docx
```

## How to Reproduce the Full Assignment

From the repository root:

```bash
python3 part1_clustering.py 10 50
python3 part1_clustering.py 15 100
python3 part1_clustering.py 5 200
python3 part2_websearch.py
python3 part3_pagerank.py
```

To run the notebook:

```bash
jupyter notebook CSL7110_Assignment4.ipynb
```

Select the kernel:

```text
Python (Assignment 4)
```

## Notes on Portability

All scripts use relative paths resolved from the script directory.

Default expected layout:

```text
spambase.data
actions.txt
answers.txt
webpages/
graphs/
```

Environment variables may also be used to override paths:

```text
SPAMBASE_PATH
WEBSEARCH_DIR
GRAPHS_DIR
```

## Final Deliverables

This repository contains:

- Source code for all three parts
- Executed Jupyter notebook
- Final PDF report
- Word version of the report
- Required datasets
- Webpage files
- Graph files
- Output validation files

## Repository Link

```text
https://github.com/Akshaykumarky26/CSL7110_Assignment4
```
