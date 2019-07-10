using Distributions
using BirthDeathProcesses
using PhyloTrees
using DataFrames
using ForwardDiff
using Optim

t = LabeledTree(read_nw(s)[1:2]...)
d = DLModel(t, 0.2, 0.3)
x = [2, 3, 4, 2]
X = [4 2 4 3; 3 1 4 3; 1 3 1 1; 4 2 1 5; 3 2 3 2]
M = get_M(d, x)
M_ = get_M(d, X)
W = get_wstar(d, M)
W_ = get_wstar(d, M_)

s = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
df = DataFrame(:A=>[2],:B=>[3],:C=>[4],:D=>[2])
df = DataFrame(:A=>[4,3,1,4,3],:B=>[2,1,3,2,2],:C=>[4,4,1,1,3],:D=>[3,3,1,5,2])
S = SpeciesTree(read_nw(s)[1:2]...)
d = DLModel(S, 0.2, 0.3)
M = profile(d, df)
W = get_wstar(d, M)
asvector(d)

gradient(d, M) |> show

set_constantrates!(S)
d = DLModel(S, postorder(S), [LinearBDP(0.2, 0.3)], Geometric(0.9))
gradient(d, M) |> show

# Naive truncated probabilistic graphical model (VE) approach (CAFE?), intuitive
function pgm(d::DLModel, x::Vector{Int64}, max=50)
    P = zeros(length(d), max+1)
    for e in d.porder
        if isleaf(d, e)
            P[e, leafmap(d[e])+1] = 1.0
        else
            children = childnodes(d, e)
            for i = 0:max
                p = 1.
                for c in children
                    p_ = 0.
                    for j in 0:length(P[c, :])-1
                        p_ += tp(d.b[c], i, j, parentdist(d, c)) * P[c, j+1]
                    end
                    p *= p_
                end
                P[e, i+1] = p
            end
        end
    end
    return P
end
