module Beluga
    using PhyloTree
    using Parameters
    using StatsFuns
    using Distributions
    using DistributedArrays
    using DataFrames
    using Random
    using StatsBase
    using AdaptiveMCMC
    using LinearAlgebra
    using MCMCChains
    using Printf
    using ForwardDiff

    import Distributions: logpdf, logpdf!, pdf
    import AdaptiveMCMC: ProposalKernel, DecreaseλProposal

    include("node.jl")
    include("model.jl")
    include("utils.jl")
    include("profile.jl")
    include("priors.jl")
    include("rjmcmc.jl")
    include("sim.jl")
    include("post.jl")

    export DuplicationLossWGDModel, DLWGD, logpdf, logpdf!, update!
    export insertwgd!, removewgd!, isawgd, iswgd, isroot, iswgdafter
    export nonwgdchild, nonwgdparent, parentdist, ne
    export Profile, PArray, extend!, shrink!, set!, rev!
    export RevJumpChain, RevJumpPrior, move!, init!, trace!
    export move_addwgd!, move_rmwgd!, branchrates!, mcmc!, rjmcmc!
    export IidRevJumpPrior, CoevolRevJumpPrior
    export branch_bayesfactors, UpperBoundedGeometric, get_wgdtrace
    export Branch, Node
end
