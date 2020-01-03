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
    import AdaptiveMCMC: ProposalKernel, DecreaseProposal

    include("node.jl")
    include("model.jl")
    include("utils.jl")
    include("profile.jl")
    include("priors.jl")
    include("rjmcmc.jl")
    include("sim.jl")
    include("post.jl")

    export DLWGD, Profile, PArray
    export RevJumpChain, IRRevJumpPrior, BMRevJumpPrior, PostPredSim
    export addwgd!, removewgd!, addwgds!, removewgds!, setrates!, getrates
    export logpdf, logpdf!, gradient, asvector, treelength
    export init!, mcmc!, rjmcmc!, wgdtrace, bayesfactors
    export posteriorΣ!, posteriorE!, pppvalues
end
