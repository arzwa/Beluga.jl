# Beluga.jl

Beluga is a julia library for the statistical analysis of gene family evolution
using phylogenetic birth-death processes. It's somewhat related to
[Whale.jl](https://arzwa.github.io/Whale.jl/dev/index.html) as it implements
models of duplication loss and whole-genome duplication, but employs gene count
data instead of gene trees. The library implements the MCMC sampler of
Zwaenepoel & Van de Peer (2019) as well as the reversible-jump MCMC sampler of
Zwaenepoel & Van de Peer (2020, *in preparation*).


## Data preparation

To perform analyses with Beluga, you will need  

1. A species tree with branch lengths (preferbly an ultrametric time tree)
2. A *phylogenetic profile matrix*. If you have a bunch of protein fasta files
   for a set of species of interest, this can be easily obtained using e.g.
   [OrthoFinder](https://github.com/davidemms/OrthoFinder).

!!! note
    If you only need the phylogenetic profile matrix from OrthoFinder, be
    sure to use the `-og` flag to stop the OrthoFinder pipeline after orthogroup
    inference. The phylogenetic profile matrix can be found in the
    `Orthogroups.GeneCount.tsv` file.
