using Test, Beluga, PhyloTrees, Parameters
import Beluga: _logpdf

# example
t, x = Beluga.example_data1()
λ = repeat([0.2], 7)
μ = repeat([0.3], 7)
d1 = DuplicationLossWGD(t, λ, μ, Float64[], 1/1.5, maximum(x))
d2 = DuplicationLossWGD(t, λ .+ 0.3, μ .+ 0.2, Float64[], 1/1.5, maximum(x))

@testset "Transition probabilities (W matrix) (DL)" begin
    W = d1.value.W
    @test W[2,2,2] ≈ 0.02311089901980
    @test W[3,4,2] ≈ 0.00066820098326
    @test W[3,4,3] ≈ 0.00433114903942
    @test W[3,4,4] ≈ 0.00314674241518
    @test W[3,4,6] ≈ 0.01493983800305
    @test W[1,1,2] ≈ 1.
end

@testset "Extinction (ϵ) probabilities (DL)" begin
    e = findroot(d1.tree)
    f, g = childnodes(d1.tree, e)
    ϵ = d1.value.ϵ
    @test ϵ[e, 2] ≈ 0.817669337
    @test ϵ[f, 1] ≈ 0.938284828
    @test ϵ[g, 1] ≈ 0.871451091
end

@testset "Conditional survival likelihood (DL)" begin
    # Verified with WGDgc (14/09/2019)
    @unpack l, L = _logpdf(d1, x)
    L = log.(L)
    shouldbe = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041,
        -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end

    @unpack l, L = _logpdf(d2, x)
    L = log.(L)
    shouldbe = [-Inf, -14.2567, -12.2106, -12.1716, -13.1142, -14.6558,
        -16.6905, -19.1649, -22.0664, -25.4208, -29.3175, -34.0229]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
end

@testset "Partial recomputation (DL)" begin
    for i in d1.tree.order
        d = deepcopy(d1)
        @unpack l, L = _logpdf(d1, x)  # get 'precomputed' L matrix
        d[:λ, i] = rand()
        d_ = DuplicationLossWGD(t, d.λ, d.μ, d.q, d.η, maximum(x))
        @test all(d.value.ϵ .== d_.value.ϵ)
        @test all(d.value.W .== d_.value.W)
        l1 = logpdf!(L, d, x, t.pbranches[i])  # partial recompute
        l2 = _logpdf(d_, x)         # compute from scratch
        @test l1 == l2.l
    end
end
