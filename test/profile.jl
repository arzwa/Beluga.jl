using PhyloTrees, CSV, Distributed, Test, DataFrames
@everywhere using DistributedArrays, Beluga

df = CSV.read("test/data/counts1.tsv", delim="\t"); deletecols!(df, :Orthogroup)
tree = SpeciesTree("test/data/tree1.nw")

@testset "PArray (single core)" begin
    p, m = Profile(df, tree)
    d = DuplicationLoss(tree, rand(17), rand(17), 0.8, m)
    @time l1 = logpdf!(d, p)
    d[:λ, 5] = 0.8
    @time l2 = logpdf!(d, p, 5)
    d_ = DuplicationLoss(tree, d.λ, d.μ, 0.8, m)
    l3 = logpdf!(d_, p)
    @test l1 != l2
    @test l2 ≈ l3
end

addprocs(2)
@everywhere using DistributedArrays, Beluga

@testset "PArray (distributed)" begin
    p, m = Profile(df, tree)
    d = DuplicationLoss(tree, rand(17), rand(17), 0.8, m)
    @time l1 = logpdf!(d, p)
    d[:λ, 5] = 0.8
    @time l2 = logpdf!(d, p, 5)
    d_ = DuplicationLoss(tree, d.λ, d.μ, 0.8, m)
    l3 = logpdf!(d_, p)
    @test l1 != l2
    @test l2 ≈ l3
end

rmprocs(workers())


function benchmark
