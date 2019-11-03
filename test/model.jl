using Test, Beluga

function example_data()
    s = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
    x = DataFrame(:A=>[2],:B=>[2],:C=>[3],:D=>[4])
    d, x = DuplicationLossWGDModel(s, x, 0.2, 0.3, 1/1.5)
    d, x[1,:]
end

@testset "DL model, Csuros & Miklos algorithm" begin
    d, x = example_data()
    L = csuros_miklos(x, d[1])
    shouldbe = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041, -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
end

@testset "DL+WGD model, Csuros & Miklos + Rabier algorithm" begin
    d, x = example_data()
    n = d[3]
    insertwgd!(d, n, 2., 0.5)
    x = [x ; [x[n.i], x[n.i]]]
    L = csuros_miklos(x, d[1])
    shouldbe = [-Inf, -12.6372, -10.112, -8.97727, -8.59388, -8.72674,
        -9.29184, -10.2568, -11.6216, -13.4219, -15.753, -18.8843]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
end

@testset "DL model, extended to DL+WGD" begin
    d, x = example_data()
    L = csuros_miklos(x, d[1])
    shouldbe = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041, -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
    n = d[3]
    insertwgd!(d, n, 2., 0.5)
    x = [x ; [x[n.i], x[n.i]]]
    L = [L ; minfs(eltype(L), 2, size(L)[2])]
    recompute!(L, x, n)
    shouldbe = [-Inf, -12.6372, -10.112, -8.97727, -8.59388, -8.72674, -9.29184, -10.2568, -11.6216, -13.4219, -15.753, -18.8843]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
end

@testset "Change λ, μ" begin
    d, x = example_data()
    for n in postwalk(d)

    L = csuros_miklos(x, d[1])
    shouldbe = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041, -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
    n = d[3]
    insertwgd!(d, n, 2., 0.5)
    x = [x ; [x[n.i], x[n.i]]]
    L = [L ; minfs(eltype(L), 2, size(L)[2])]
    recompute!(L, x, n)
    shouldbe = [-Inf, -12.6372, -10.112, -8.97727, -8.59388, -8.72674, -9.29184, -10.2568, -11.6216, -13.4219, -15.753, -18.8843]
    for i=1:length(L[1,:])
        @test isapprox(L[1, i] , shouldbe[i], atol=0.0001)
    end
end



@testset "Root posterior with geometric prior (DL+WGD)" begin
    d = deepcopy(d2)
    @unpack l, L = _logpdf(d, x)
    @test isapprox(l, -11.3952503756417, atol=0.00001)
    d.η = 0.9
    l = logpdf!(L, d, x, [1])
    @test isapprox(l, -11.4734876141945, atol=0.00001)
    for i=1:10
        d.η = rand()
        d_ = DuplicationLossWGD(t, d.λ, d.μ, d.q, d.η, maximum(x))
        l = logpdf!(L, d, x, [1])
        l_ = _logpdf(d_, x)
        @test l ≈ l_.l
    end
end
