using Test
using PhyloTrees
using DataFrames
import Beluga: csuros_miklos, integrate_root

s = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
t = SpeciesTree(read_nw(s)[1:2]...)
d = DLModel(t, 0.2, 0.3)
x = DataFrame(:A=>[2],:B=>[2],:C=>[3],:D=>[4])
M = profile(t, x)
W = Beluga.get_wstar(d, M)

@testset "Extended profile" begin
    @test M[1,:] == [11, 4, 7, 3, 4, 2, 2]
end

@testset "Extinction probabilities" begin
    e = findroot(t)
    f, g = childnodes(d, e)
    @test d.ϵ[e, 2] ≈ 0.817669337
    @test d.ϵ[f, 1] ≈ 0.938284828
    @test d.ϵ[g, 1] ≈ 0.871451091
end

@testset "Transition probabilities (w[.|.])" begin
    @test W[:,:,5][2,2] ≈ 0.14591589724
    @test W[:,:,5][3,5] ≈ 0.00407844288
    @test W[:,:,2][3,5] ≈ 0.00062696284
    @test W[:,:,2][2,2] ≈ 0.02311089902
    @test W[:,:,2][6,2] ≈ 0.0
end

@testset "Log-likelihood ~ Csuros & Miklos" begin
    shouldbe = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041,
        -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
    L = csuros_miklos(d, M[1,:], W)[1,:]
    for i in eachindex(L)
        @test isapprox(L[i], shouldbe[i], atol=1e-4)
    end
end

@testset "Integrate over # root lineages: Geometric" begin
    L = csuros_miklos(d, M[1,:], W)
    @test integrate_root(L[1,:], d.ρ, d, 1) ≈ -14.38233786
end
