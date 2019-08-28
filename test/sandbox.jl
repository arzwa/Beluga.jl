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
df = DataFrame(:A=>[4,3,1,4,3],:B=>[2,1,3,2,2],:C=>[4,4,1,1,3],:D=>[3,3,1,5,2])
S = SpeciesTree(read_nw(s)[1:2]...)
d = DLModel(S, 0.2, 0.3)
M = profile(d, df)
W = Beluga.get_wstar(d, M)
asvector(d)

Beluga.gradient(d, M) |> show

set_constantrates!(S)
d = DLModel(S, 0.2, 0.3, 0.9)
gradient(d, M) |> show

d = DLModel(s, 0.002, 0.003)
d, out = mle(d, M[2, :])
