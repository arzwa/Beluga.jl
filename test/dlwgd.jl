using Test, Beluga, PhyloTrees

t, x = Beluga.example_data2()
λ = repeat([0.2], 7)
μ = repeat([0.3], 7)
d1 = DuplicationLossWGD(t, λ, μ, [0.5], 1/1.5, maximum(x))
d2 = DuplicationLossWGD(t, λ .+ 0.3, μ .+ 0.2, [0.2], 1/1.5, maximum(x))

@testset "Extinction (ϵ) probabilities (DL+WGD)" begin
    # tested with WGDgc (14/09/2019)
    e = findroot(d1.tree)
    f, g = childnodes(d1.tree, e)
    ϵ = d1.value.ϵ
    @test ϵ[e, 2] ≈ 0.778372527869781
    @test ϵ[f, 1] ≈ 0.938284827880156
    @test ϵ[g, 1] ≈ 0.829569555790793
end

@testset "Transition probabilities (W matrix) (DL+WGD)" begin
    # tested with WGDgc (14/09/2019)
    W = d1.value.W
    shouldbe = [0.0, 0.000668201, 0.00671049, 0.0133832, 0.00754552,
        0.00314674, 0.0107602, 0.0149398, 0.0149398]
    for i=1:length(shouldbe)
        @test isapprox(W[3,4,i], shouldbe[i], atol=1e-5)
    end
end

@testset "Conditional survival likelihood (DL+WGD)" begin
    # Verified with WGDgc (16/09/2019)
    l, L = logpdf(d1, x)
    L = log.(L)
    shouldbe = [-Inf, -12.6372, -10.112, -8.97727, -8.59388, -8.72674,
        -9.29184, -10.2568, -11.6216, -13.4219, -15.753, -18.8843]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end

    l, L = logpdf(d2, x)
    L = log.(L)
    shouldbe = [-Inf, -14.2131, -12.2466, -12.2982, -13.3171, -14.931,
        -17.0352, -19.5772, -22.5452, -25.9652, -29.9266, -34.6961]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
end

@testset "Partial recomputation (DL+WGD)" begin
    for i in d1.tree.order
        d = deepcopy(d1)
        l, L = logpdf(d1, x)  # get 'precomputed' L matrix
        d[:λ, i] = rand()
        d_ = DuplicationLossWGD(t, d.λ, d.μ, d.q, d.η, maximum(x))
        @test all(d.value.ϵ .== d_.value.ϵ)
        @test all(d.value.W .== d_.value.W)
        bs = Beluga.get_parentbranches(t, i)
        l1 = logpdf!(L, d, x, bs)  # partial recompute
        l2, L2 = logpdf(d_, x)     # compute from scratch
        @test l1 == l2
    end
end

@testset "Root posterior with geometric prior (DL+WGD)" begin
    d = deepcopy(d2)
    l, L = logpdf(d, x)
    @test isapprox(l, -11.3952503756417, atol=0.00001)
    d[:η] = 0.9
    l = logpdf!(L, d, x, [1])
    @test isapprox(l, -11.4734876141945, atol=0.00001)
    for i=1:10
        d[:η] = rand()
        d_ = DuplicationLossWGD(t, d.λ, d.μ, d.q, d.η, maximum(x))
        l = logpdf!(L, d, x, [1])
        l_, L_ = logpdf(d_, x)
        @test l ≈ l_
    end
end
