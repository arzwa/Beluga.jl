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
    using RecipesBase

    import Distributions: logpdf, logpdf!, pdf
    import AdaptiveMCMC: ProposalKernel, DecreaseProposal

    include("node.jl")
    include("model.jl")
    include("utils.jl")
    include("profile.jl")
    include("priors.jl")
    include("rjmcmc.jl")
    include("sim.jl")
    include("post.jl")

    export DuplicationLossWGDModel, DLWGD, logpdf, logpdf!
    export insertwgd!, removewgd!, Profile, PArray
    export RevJumpChain, RevJumpPrior, branchrates!, mcmc!, rjmcmc!
    export IidRevJumpPrior, CoevolRevJumpPrior, extend!, shrink!
    export branch_bayesfactors, UpperBoundedGeometric, get_wgdtrace
    export Branch, Node, asvector, gradient, init!, update!
    export posterior_Σ!, posterior_E!, PostPredSim, pp_pvalues, treelength
end
