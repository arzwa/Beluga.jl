
begin
    nw = readline("example/9dicots/9dicots.nw")
    df = CSV.read("example/9dicots/9dicots-f01-100.csv")
    d, p = DLWGD(nw, df, 2., 2., 0.9)
    # d, p = DLWGD(nw, 2., 2., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([1,1]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,3),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d),
        πE=Normal(1,0.05))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    props = Beluga.AMMProposals(2*Beluga.ne(d)+2, σ=0.1, β=0.05)
    chain = RevJumpChain(data=p, model=d, prior=prior,
        kernel=kernel, props=props, da=true)
    init!(chain)
    for i=1:100
        @time rjmcmc!(chain, 100, trace=1, show=10)
        p = plot([plot(chain.trace[!,Symbol("λ$i")]) for i=1:4:16]...,
                 [stephist(log10.(chain.trace[!,Symbol("λ$i")]),
                    fill=true, normalize=true, alpha=0.5) for i=1:4:16]...,
                  plot(chain.trace[!,:k]),
                 legend=false, grid=false)
        display(p)
    end
end

begin
    nw = readline("example/9dicots/9dicots.nw")
    df = CSV.read("example/9dicots/9dicots-f01-100.csv")
    d, p = DLWGD(nw, df, 2., 2., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([1,1]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    chain2 = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel, da=true)
    init!(chain2)
    for i=1:100
        @time rjmcmc!(chain2, 100, trace=1, show=10)
    end
end

begin
    nw = readline("example/9dicots/9dicots.nw")
    df = CSV.read("example/9dicots/9dicots-f01-100.csv")
    d, p = DLWGD(nw, df, 2., 2., 0.9)
    prior = Beluga.CRRevJumpPrior(
        X₀=MvNormal(log.([1,1]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.SimpleKernel(qkernel=Beta(1,3))
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel, da=true)
    init!(chain)
    for i=1:100
        @time rjmcmc!(chain, 100, trace=1, show=10)
    end
end

#     DA: 143.103991 seconds (501.90 M allocations: 107.977 GiB, 10.66% gc time)
#  no DA: 189.500442 seconds (796.78 M allocations: 169.154 GiB, 10.63% gc time)
# DA+AMM: 103.014576 seconds (350.40 M allocations: 72.692 GiB, 15.11% gc time)
# BUT: pjump(DA) << pjump(no-DA) and average prior DA > no-DA
