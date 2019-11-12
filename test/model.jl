# Main model tests
# ================
using Test, DataFrames, CSV, Distributions, Random
import Beluga: csuros_miklos, csuros_miklos!, minfs
Random.seed!(333)

df1 = DataFrame(:A=>[2,3],:B=>[2,5],:C=>[3,0],:D=>[4,1])
df2 = DataFrame(:A=>rand(1:20, 1000),:B=>rand(1:20, 1000),:C=>rand(1:10, 1000),:D=>rand(1:10, 1000))
df3 = CSV.read("data/plants1-100.tsv", delim=",")

s1 = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
s3 = open("data/plants1.nw", "r") do f ; readline(f); end

shouldbe1 = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041, -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
shouldbe2 = [-Inf, -12.6372, -10.112, -8.97727, -8.59388, -8.72674, -9.29184, -10.2568, -11.6216, -13.4219, -15.753, -18.8843]


@testset "DL model, Csuros & Miklos algorithm" begin
    d, y = DuplicationLossWGDModel(s1, df1, 0.2, 0.3, 1/1.5)
    x = y[:,1]
    L = csuros_miklos(x, d[1])
    for i=1:length(L[:,1])
        @test isapprox(L[i, 1] , shouldbe1[i], atol=0.0001)
    end
end


@testset "DL+WGD model, Csuros & Miklos + Rabier algorithm" begin
    d, y = DuplicationLossWGDModel(s1, df1, 0.2, 0.3, 1/1.5)
    x = y[:,1]
    n = d[3]
    insertwgd!(d, n, 2., 0.5)
    x_ = [x ; [x[n.i], x[n.i]]]
    L = csuros_miklos(x_, d[1])
    for i=1:length(L[:,1])
        @test isapprox(L[i, 1] , shouldbe2[i], atol=0.0001)
    end
end


@testset "DL model, extend/remove to/from DL+WGD" begin
    # compute lhood, add WGD, partially recompute, remove WGD, recompute again
    d, y = DuplicationLossWGDModel(s1, df1, 0.2, 0.3, 1/1.5)
    x = y[:,1]
    L = csuros_miklos(x, d[1])
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe1[i], atol=0.0001)
    end

    n = d[3]
    wgdnode = insertwgd!(d, n, 2., 0.5)
    x_ = [x ; [x[n.i], x[n.i]]]
    L = [L minfs(eltype(L), size(L)[1], 2)]
    logpdf!(L, x_, n)
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe2[i], atol=0.0001)
    end

    child = removewgd!(d, wgdnode)
    logpdf!(L, x_, child)
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe1[i], atol=0.0001)
    end
end


@testset "Partial recomputation (1)" begin
    d, y = DuplicationLossWGDModel(s1, df1, 0.2, 0.3, 1/1.5)
    x = y[:,1]
    n = d[3]
    update!(n, :λ, 0.9)
    L = csuros_miklos(x, d[1])
    for i=1:length(L[:,1])
        @test L[i,1] != shouldbe1[i] || L[i,1] == -Inf
    end
    update!(n, :λ, 0.2)
    while !isnothing(n)
        csuros_miklos!(L, x, n)
        n = n.p
    end
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe1[i], atol=0.0001)
    end
end


@testset "Partial recomputation (2)" begin
    d, y = DuplicationLossWGDModel(s3, df3, 0.2, 0.3, 1/1.5)
    p = Profile(y)
    l1 = logpdf!(d, p)
    update!(d[5], :λ, 0.67)
    l2 = logpdf!(d, p)
    @test l1 != l2
    update!(d[3], :μ, 0.12)
    l3 = logpdf!(d, p)
    @test l1 != l2 != l3
    update!(d[1], (η=0.4, λ=0.1))
    l4 = logpdf!(d, p)
    @test l1 != l2 != l3 != l4
    update!(d[5], :λ, 0.2)
    update!(d[3], :μ, 0.3)
    update!(d[1], (η=1/1.5, λ=0.2))
    l5 = logpdf!(d, p)
    @test l5 == l1
end


@testset "Insert and remove WGDs" begin
    d, y = DuplicationLossWGDModel(s1, df1, 0.2, 0.3, 1/1.5)
    insertwgd!(d, d[3], 2., 0.2)
    insertwgd!(d, d[3], 1., 0.3)
    insertwgd!(d, d[2], 1., 0.4)
    removewgd!(d, d[8])
    @test d[8][:q] == 0.3
    @test d[10][:q] == 0.4
    removewgd!(d, d[10])
    @test d[8][:q] == 0.3
    @test !haskey(d[9].x.θ, :q)
    @test !haskey(d.nodes, 10)
    @test !haskey(d.nodes, 11)
    x = y[:,1]
    @test logpdf(d, [x; [x[3], x[3]]]) ≈ -9.159735036143305
end


@testset "Stress test (shouldn't error)" begin
    # shouldn't error
    r1 = Normal(0, 5)
    r2 = Beta(3, 1/3)
    for i=1:100
        λ, μ, η = exp(rand(r1)), exp(rand(r1)), rand(r2)
        d, y = DuplicationLossWGDModel(s1, df1, λ, μ, η)
        logpdf(d, y[:,1])
    end
    r1 = Normal(0, 5)
    for i=1:100
        λ, η = exp(rand(r1)), rand(r2)
        d, y = DuplicationLossWGDModel(s1, df1, λ, λ, η)
        logpdf(d, y[:,1])
    end
end
