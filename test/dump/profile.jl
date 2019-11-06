using PhyloTrees, CSV, Distributed, Test, DataFrames
addprocs(2)
@info "Added procs, now using $(nworkers()) processes"
@everywhere using DistributedArrays, Beluga

df = CSV.read("data/hexapods1-10.tsv", delim="\t"); deletecols!(df, :Orthogroup)
tree = SpeciesTree("data/hexapods1.nw")

@testset "PArray (distributed; 2 workers)" begin
    p, m = Profile(df, tree)
    d = DuplicationLoss(tree, rand(17), rand(17), 0.8, m)
    l1 = logpdf!(d, p)
    d[:λ, 5] = 0.8
    l2 = logpdf!(d, p, 5)
    d_ = DuplicationLoss(tree, d.λ, d.μ, 0.8, m)
    l3 = logpdf!(d_, p)
    @test l1 != l2
    @test l2 ≈ l3
end

@testset "PArray partial recomputation etc." begin
    p, m = Profile(df, tree)
    d = DuplicationLoss(tree, rand(17), rand(17), 0.8, m)
    l1 = logpdf!(d, p)
    set_L!(p)

    # changing η should leave matrix intact, but change lhood
    d.η = rand()
    l2 = logpdf!(d, p, 1)
    @test l2 != l1
    for i in 1:length(p)
        @test all(p[i].Ltmp .== p[i].L)
    end

    # this should properly test partial recomputation in the PArray
    for e in tree.order[1:end-1]
        d[:λ, e] = rand()
        l3 = logpdf!(d, p, e)
        for b in tree.order
            if !(b in tree.pbranches[e]) || b == e
                # NOTE: node `e` lhood not affected by branch `e` param change!
                for i=1:length(p)
                    @test all(p[i].Ltmp[b,:] .== p[i].L[b,:])
                end
            else
                for i=1:length(p)
                    @test ((any(p[i].Ltmp[b,:] .!= p[i].L[b,:])) ||
                        p[i].Ltmp[b,1] == p[i].L[b,1] == 0.)
                    # second condition captures case of leaf count zero
                end
            end
        end
        set_L!(p)
    end
end

rmprocs(workers())
