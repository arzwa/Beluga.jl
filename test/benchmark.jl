using Distributed
using DataFrames, CSV, PhyloTree

nmax = 4
nrep = 100
res = []
for n = 0:1:nmax
    addprocs(n)
    @everywhere using Pkg
    @everywhere Pkg.activate("/home/arzwa/dev/Beluga/")
    @everywhere using Beluga
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, df, 1., 1., 0.9, Beluga.Branch)
    logpdf!(d, p)
    ts = [@timed(logpdf!(d, p))[2] for i=1:nrep]
    push!(res, (nworkers(), mean(ts), std(ts)))
    rmprocs(workers())
end

println("workers, mean_t, std_t")
for rep in res
    println(join(round.(rep, digits=4), ", "))
end
