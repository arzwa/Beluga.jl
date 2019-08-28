using CSV
using DataFrames
using Beluga

s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts1.csv", delim="\t"); deletecols!(df, :Orthogroup)
M = profile(s, df)

@testset "Single family" begin
    s_ = deepcopy(s)
    Beluga.set_constantrates!(s_)
    d = DLModel(s_, 0.002, 0.003)
    d, out = mle(d, M[2, :], show_trace=false)
    @test d[1].λ ≈ 0.18368227181154387
    @test d[1].μ ≈ 0.09288814282687102
    d, out = mle(d, M[4, :],  show_trace=false)
    @test d[1].λ ≈ 0.0398651674431491
    @test d[1].μ ≈ 1.0823504234103797e-8
end

@testset "Bunch of families, constant rates" begin
    s_ = deepcopy(s)
    Beluga.set_constantrates!(s_)
    d = DLModel(s_, 0.002, 0.003)
    d, out = mle(d, M, show_trace=false)
end

@testset "Bunch of families, branch-wise rates" begin
    d = DLModel(s, 0.002, 0.003)
    d, out = mle(d, M, show_trace=false)
end
