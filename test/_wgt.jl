using Beluga, CSV, Distributions

tree = readline("example/dicots/dicots.nw")
data = CSV.read("example/dicots/dicots-f01-25.csv")
wgds = [(lca="ath", t=rand(), q=rand()),
        (lca="ath", t=rand(), q=rand()),
        (lca="ptr", t=rand(), q=rand()),
        (lca="mtr", t=rand(), q=rand()),
        (lca="cqu", t=rand(), q=rand()),
        (lca="ugi", t=rand(), q=rand()),
        (lca="ugi", t=rand(), q=rand()),
        (lca="ugi", t=rand(), q=rand()),
        (lca="sly", t=rand(), q=rand(), l=3)]

m, p = DLWGD(tree, data)
addwgds!(m, p, wgds)

prior = IRRevJumpPrior(
    Ψ=[1 0.0 ; 0.0 1],
    X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
    πK=DiscreteUniform(0, 10),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=Beluga.treelength(m))
kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
chain = RevJumpChain(data=p, model=m, prior=prior, kernel=kernel)
init!(chain)
mcmc!(chain, 2000, trace=1, show=1)
