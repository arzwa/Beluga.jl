# Main model tests
# ================
import Beluga: csuros_miklos, csuros_miklos!, minfs, update!


@testset "DL model, Csuros & Miklos algorithm" begin
    d, p = DLWGD(s1, df1, 0.2, 0.3, 1/1.5, Node)
    L = csuros_miklos(d[1], p[1].x)
    for i=1:length(L[:,1])
        @test isapprox(L[i, 1] , shouldbe1[i], atol=0.0001)
    end
end


@testset "DL+WGD model, Csuros & Miklos + Rabier algorithm" begin
    d, p = DLWGD(s1, df1, 0.2, 0.3, 1/1.5, Node)
    x = p[1].x
    n = d[3]
    insertwgd!(d, n, 2., 0.5)
    x_ = [x ; [x[n.i], x[n.i]]]
    L = csuros_miklos(d[1], x_)
    for i=1:length(L[:,1])
        @test isapprox(L[i, 1] , shouldbe2[i], atol=0.0001)
    end
end


@testset "DL model, extend/remove to/from DL+WGD" begin
    # compute lhood, add WGD, partially recompute, remove WGD, recompute again
    d, p = DLWGD(s1, df1, 0.2, 0.3, 1/1.5, Node)
    x = p[1].x
    L = csuros_miklos(d[1], x)
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe1[i], atol=0.0001)
    end

    n = d[3]
    wgdnode = insertwgd!(d, n, 2., 0.5)
    x_ = [x ; [x[n.i], x[n.i]]]
    L = [L minfs(eltype(L), size(L)[1], 2)]
    logpdf!(L, n, x_)
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe2[i], atol=0.0001)
    end

    child = removewgd!(d, wgdnode)
    logpdf!(L, child, x_)
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe1[i], atol=0.0001)
    end
end


@testset "Partial recomputation (1)" begin
    d, p = DLWGD(s1, df1, 0.2, 0.3, 1/1.5, Node)
    x = p[1].x
    n = d[3]
    update!(n, :λ, 0.9)
    L = csuros_miklos(d[1], x)
    for i=1:length(L[:,1])
        @test L[i,1] != shouldbe1[i] || L[i,1] == -Inf
    end
    update!(n, :λ, 0.2)
    while !isnothing(n)
        csuros_miklos!(L, n, x)
        n = n.p
    end
    for i=1:length(L[:,1])
        @test isapprox(L[i,1] , shouldbe1[i], atol=0.0001)
    end
end


@testset "Partial recomputation (2)" begin
    d, p = DLWGD(s3, df3, 0.2, 0.3, 1/1.5, Node)
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
    d, p = DLWGD(s1, df1, 0.2, 0.3, 1/1.5, Node)
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
    x = p[1].x
    @test logpdf(d, [x; [x[3], x[3]]]) ≈ -9.159735036143305
end


@testset "Stress test (shouldn't error)" begin
    # shouldn't error
    r1 = Normal(0, 5)
    r2 = Beta(3, 1/3)
    for i=1:100
        λ, μ, η = exp(rand(r1)), exp(rand(r1)), rand(r2)
        d, p = DLWGD(s1, df1, λ, μ, η, Node)
        logpdf!(d, p)
    end
    r1 = Normal(0, 5)
    for i=1:100
        λ, η = exp(rand(r1)), rand(r2)
        d, p = DLWGD(s1, df1, λ, λ, η, Node)
        logpdf!(d, p)
    end
end


@testset "Profile DL model, full loglikelihood" begin
    # test data based on previous Beluga implementation
    λ = [0.56, 0.97, 0.12, 0.21, 4.23, 0.12, 0.77, 2.01, 2.99, 0.44]
    μ = [0.64, 0.36, 0.25, 0.57, 0.42, 0.49, 0.61, 0.69, 3.92, 0.53]
    η = [0.55, 0.56, 0.93, 0.57, 0.59, 0.06, 0.15, 0.41, 0.96, 0.12]
    for i=1:length(λ)
        d, p = DLWGD(s3, df3, λ[i], μ[i], η[i], Node)
        l = logpdf!(d, p)
        @test shouldbe3[i] ≈ l
    end
end


@testset "Profile DL model, extending and shrinking" begin
    # test data based on previous Beluga implementation
    λ = [0.56, 0.97, 0.12, 0.21, 4.23, 0.12, 0.77, 2.01, 2.99, 0.44]
    μ = [0.64, 0.36, 0.25, 0.57, 0.42, 0.49, 0.61, 0.69, 3.92, 0.53]
    η = [0.55, 0.56, 0.93, 0.57, 0.59, 0.06, 0.15, 0.41, 0.96, 0.12]
    for i=1:length(λ)
        d, p = DLWGD(s3, df3, λ[i], μ[i], η[i], Node)
        l = logpdf!(d, p)
        r = rand(2:length(d), 10)
        wnodes = []
        while length(wnodes) != length(r) || length(r) != 0
            u = rand()
            if length(r) > 0 && u < 0.5
                j = pop!(r)
                w = insertwgd!(d, d[j], rand()*d[j][:t], rand())
                extend!(p, j)
                logpdf!(d[j], p)
                push!(wnodes, w)
            elseif length(wnodes) > 0 && u > 0.5
                wgdnode = pop!(wnodes)
                child = removewgd!(d, wgdnode)
                shrink!(p, wgdnode.i)
                logpdf!(child, p)
            end
        end
        # @show length(d), length(postwalk(d[1]))
        l = logpdf!(d, p)
        @test shouldbe3[i] ≈ l
    end
end


@testset "Gradient (profile)" begin
    g = [0.0, 0.21654, -4.29457, 0.28282, -0.07077, -1.77586, -2.57598,
        6.25627, -2.82879, 1.81408, 3.84539, 19.34362, -0.47348, -1.39255,
        -2.83181, 27.41048, -1.44903, -2.66647, 3.29678, 0.0, -0.08539,
        2.6401, -0.37875, 0.12269, -1.05525, -1.48215, -1.38917, 7.49706,
        0.25451, -2.1514, -10.16407, 0.06406, 1.94643, 11.01334, -1.90087,
        1.26579, 22.01815, -3.40453, 49.2567]
    d, p = DLWGD(s4, df3, 2., 1., 0.9, Branch)
    g_ = Beluga.gradient(d, p)
    for i=1:length(g)
        @test isapprox(g[i], g_[i], atol=0.0001)
    end
end
