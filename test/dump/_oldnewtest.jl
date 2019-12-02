using Pkg
Pkg.activate("/home/arzwa/dev/Beluga/")
using Beluga
using Test, DataFrames, CSV, Distributions, LinearAlgebra

# begin  # old
#     df = CSV.read("test/data/plants1-100.tsv", delim=",")
#     nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
#     d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9, Beluga.BelugaBranch)
#     p = Profile(y)
#     prior = IidRevJumpPrior(
#         Σ₀=[1 0.9 ; 0.9 1],
#         X₀=MvNormal(log.([2,2]), I),
#         πK=Beluga.UpperBoundedGeometric(0.3, 15),
#         πq=Beta(1,1),
#         πη=0.9)
#     chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
#     init!(chain, rjump=(1., 10., 0.01))
#     ofile = "test/old.csv"
# end

begin  # new
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, df, 1., 1., 0.9, Beluga.Branch)
    prior = IidRevJumpPrior(
        Σ₀=[1 0.9 ; 0.9 1],
        X₀=MvNormal(log.([2,2]), I),
        πK=Beluga.UpperBoundedGeometric(0.3, 15),
        πq=Beta(1,1),
        πη=Beluga.ConstantDistribution(0.9))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, rjump=(1., 10., 0.01))
    ofile = "test/new.csv"
end

nrep = 100
open(ofile, "w") do f
    for (i,n) in sort(d.nodes)
        x = [@timed(Beluga.move_node!(chain, n)) for i=1:nrep]
        t = [xx[2] for xx in x]
        a = [xx[3] for xx in x]
        write(f, "$i,$(mean(t)),$(std(t)),$(mean(a)),$(std(a))\n")
        print("$i,$(mean(t)),$(std(t)),$(mean(a)),$(std(a))\n")
    end
end
