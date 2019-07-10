module Beluga

    using Distributions
    using BirthDeathProcesses
    using PhyloTrees

    include("dlmodel.jl")

    export
        DLModel, get_M, get_wstar, _logpdf
end
