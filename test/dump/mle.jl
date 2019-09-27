using Test
using CSV
using DataFrames
using Beluga

s = SpeciesTree("data/tree1.nw")
df = CSV.read("data/counts1.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)

@testset "Single family" begin
    s_ = deepcopy(s)
    Beluga.set_constantrates!(s_)
    d = DuplicationLoss(s_, [0.002], [0.003], 0.8, m)
    d, out = mle(d, p[2], show_trace=false)
    @test d[1].λ ≈ 0.18368227181154387
    @test d[1].μ ≈ 0.09288814282687102
    d, out = mle(d, M[4, :],  show_trace=false)
    @test d[1].λ ≈ 0.0398651674431491
    @test d[1].μ ≈ 1.0823504234103797e-8
end

@testset "Bunch of families, constant rates" begin
    s_ = deepcopy(s)
    Beluga.set_constantrates!(s_)
    d = DLModel(s_, maximum(M), 0.002, 0.003)
    d, out = mle(d, M, show_trace=false)
    @test d[1].λ ≈ 0.16113470734878857
    @test d[1].μ ≈ 0.0833231295783750
end

#=@testset "Bunch of families, branch-wise rates" begin
    d = DLModel(s, maximum(M), 0.002, 0.003)
    d, out = mle(d, M, show_trace=false)
    @test (d[4]).λ ≈ 0.0013679397375902774
    @test (d[3]).λ ≈ 1.4888627510063854
    @test (d[8]).λ ≈ 0.00020685919842907614
    @test (d[2]).μ ≈ 8.578681288777693
    @test (d[7]).μ ≈ 6.623788347647633e-7
    @test (d[9]).μ ≈ 0.2148465843386549
end=#
