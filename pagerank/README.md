# PageRank from Scratch

A small Information Retrieval project that implements PageRank using a column-stochastic transition matrix and power iteration, without calling `networkx.pagerank`.

## What it demonstrates

- Constructing a transition matrix from graph links.
- Handling dangling nodes by redistributing their probability uniformly.
- Iterating the PageRank update until the rank vector converges.
- Comparing PageRank with normalized in-degree.
- Checking that transition-matrix columns sum to one and that PageRank scores sum to one.

## Graph and parameters

The notebook builds a reproducible 30-node Barabási–Albert graph, converts its edges to a directed representation, and removes the outgoing edges of nodes 5, 12, and 25 to create dangling nodes. The default damping factor is `0.85`; power iteration stops when the Euclidean norm between successive rank vectors is below `1e-6` or after 100 iterations.

## Report and analysis

The saved notebook output reports convergence after **18 iterations**. Its numerical checks show transition-matrix column sums from `1.000000` to `1.000000` and a total PageRank score of `1.000000`.

**Q1 — What if dead ends were left as zero columns?**  
Probability mass would leak from the random-surfer process whenever rank reaches a dangling node, because that column would redistribute none of its probability. Repeated iterations could therefore reduce the total rank mass below one. Redistributing a dangling node's outgoing probability uniformly preserves the column-stochastic property.

**Q2 — What is the effect of the damping factor?**  
As `d` approaches 1, the surfer follows links more often and teleports less. The ranking becomes more dependent on graph topology, and power iteration typically converges more slowly. At `d = 0`, the link-following contribution vanishes and each update is uniform, assigning `1/N` rank to every node.

**Q3 — How can a low in-degree node have relatively high PageRank?**  
In the recorded comparison, Node 16 has PageRank `0.032525` with an in-degree share of `2.92%`. Node 25 has the same in-degree share (`2.92%`) but a lower score (`0.027811`). In this generated graph, Node 16 receives links from Nodes **4, 14, 22, and 28**, while Node 25 receives links from Nodes **0, 4, 7, and 26**. PageRank contributions depend on each linking node's rank divided by its out-degree, so links from nodes that distribute their rank across fewer outgoing edges can contribute more per link. For example, Nodes 22 and 28 have out-degrees 4 and 3, respectively. These figures describe only the synthetic graph in this notebook, not real web pages.

**Q4 — Does teleportation help a page with no incoming links?**  
Yes. Teleportation contributes `(1 - d) / N` to every page at each update, so a page can receive rank even when no other node links to it. Its final score can be higher than that contribution because redistribution from dangling nodes may also send probability to it.

## Run locally

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook PageRank_Notebook.ipynb
```

Run the notebook cells from top to bottom. No external dataset is required; the graph is generated in the notebook.

## Main implementation

- `build_stochastic_matrix(graph)` constructs the transition matrix and fixes dangling-node columns.
- `power_iteration(M, d=0.85, max_iter=100, tol=1e-6)` computes the rank vector through repeated matrix-vector multiplication.

NetworkX is used to create and visualize the graph; the PageRank calculation itself is implemented explicitly with NumPy.
