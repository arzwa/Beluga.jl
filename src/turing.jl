using Turing
using PhyloTrees
using Beluga

s = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
df = DataFrame(:A=>[4,3,1,4,3],:B=>[2,1,3,2,2],:C=>[4,4,1,1,3],:D=>[3,3,1,5,2])
S = SpeciesTree(read_nw(s)[1:2]...)
d = DLModel(S, 0.2, 0.3)
n = length(d.b)
X = profile(S, df)


@model dlmodel(x, ::Type{TV}=Vector{Float64}) where TV = begin
    θ ~ MvLogNormal([log(1.), log(1.)], [1. 0.9; 0.9 1.])
    λ ~ MvLogNormal(repeat([log(θ[1])], n), ones(n))
    μ ~ MvLogNormal(repeat([log(θ[2])], n), ones(n))
    η ~ Beta(10, 1)
    x ~ DLModel(S, λ, μ, η)
end

m = dlmodel(X)
chn = sample(m, HMC(1000, 0.1, 5))
