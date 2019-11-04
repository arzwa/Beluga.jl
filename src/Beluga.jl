module Beluga
    using PhyloTree
    using Parameters
    using StatsFuns
    using Distributions
    using DataFrames

    import Distributions: logpdf, logpdf!

    include("_model.jl")
    include("_utils.jl")
    include("_profile.jl")

    export DuplicationLossWGDModel, DLWGD, logpdf, logpdf!, update!
    export insertwgd!, removewgd!
    export Profile, PArray
end
