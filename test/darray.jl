using Distributed
using PhyloTrees
using CSV
using DataFrames
@everywhere using DistributedArrays
@everywhere using Beluga

s = SpeciesTree(readtree("test/data/tree1.nw"))
df = CSV.read("test/data/counts1.csv", delim="\t"); deletecols!(df, :Orthogroup)
d = DLModel(s, 0.2, 0.3)
n = length(d.b)
X = profile(s, df)

# some benchmarking on different data set sizes and number of CPUs for
# different approaches
for i=1:4
    addprocs(i)
    println("# CORES = $i $(repeat("-", 70))")
    @everywhere using DistributedArrays
    @everywhere using Beluga
    for n in [10, 100, 200, 500]
        @show n*10
        X_ = vcat([X for i=1:n]...)
        Y = [X_[i,:] for i=1:size(X_)[1]]
        E = distribute(Y)
        D = distribute(X_)
        println("Matrix")
        @time logpdf(d, X_)
        @time logpdf(d, X_)
        println("DMatrix")
        @time logpdf(d, D)
        @time logpdf(d, D)
        println("Array of vectors")
        @time mapreduce(x->logpdf(d, x), +, Y)
        @time mapreduce(x->logpdf(d, x), +, Y)
        println("DArray of vectors")
        @time mapreduce(x->logpdf(d, x), +, E)
        @time mapreduce(x->logpdf(d, x), +, E)
    end
    rmprocs(workers())
end
