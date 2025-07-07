# Filtering step for sPHENIX pp collision data
I've tried to mimic the embedding and filtering step of the `Exa.TrkX`
pipeline. However, based on the input from the `Exa.TrkX` experts,
the key is not follow their steps but to ensure the connectivity
before entering the GNN step. According to one of the expert, if there
is no limitation on GPU memory, they would just put the entire graph
into the GNN without filter, and let the GNN to figure out edge
classification ^_^

Although the pp collision data we have is sparse, it is probably still
not a good idea to entering GNN with no filtering at all. However,
instead of replicate the emebedding and filtering steps of `Exa.TrkX`,
we probably should set the connectivity as goal. That is, we may want
to set goals for edge classficiation efficiency and purity, and
achieve the goal in the pre-GNN step(s).

Now, let us check the paper for pre-GNN classification performance and
connectivity level.
After the Embedding step, the paper mentioned that: "The edge
selection at this stage is close to 100% efficient but O(1)% pure,
with a graph size of O(10^5) nodes and O(10^7) edges." After the
filtering step, the paper states that "Constraining edge efficiency to
remain high (above 96%) leads to much sparser graphs, of O(10^6)
edges."

We interpret this information as that we need to have > 96% efficiency
on edge prediction and a node is connected to 10 other nodes on
average.

We also need to note that the work only put an edge between immediate
neighbors. It is probably make better sense since TrackML has only 16
layers and hence lower resolution along the radial dimension. However,
we are not sure whether it is still proper choice for sPHENIX data
with 48 layers. Here is what I would do it.
1. Combine embedding and filtering (let us call it filtering);
1. Figure out immediate neighbor in pre-processing;
1. construct node features as "node itself + 2-closest neighbors"
1. Train with 3 types of pair queries:
    1. True edge (between immediate neighbors)
    2. KNN-neighborhood with radius cap with k and radius cap so chosen
        that > 99.9% immediate neighbor edge are included.
    3. random pairs.

    The three types of queries all have the same number as the true
    edges.
    I also don't want to punish misclassification of non-immediate
    neighbor as harsh as non-neighbor. I may achieve this by weighting,
    but let us keep it optional at this moment.
1. run evaluation code to choose a proper threshold that approximately
    achieves the classification and connectivity goal.
