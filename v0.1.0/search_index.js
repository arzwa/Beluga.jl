var documenterSearchIndex = {"docs":
[{"location":"api/#API-1","page":"API","title":"API","text":"","category":"section"},{"location":"api/#","page":"API","title":"API","text":"Modules = [Beluga]","category":"page"},{"location":"api/#Beluga.BMRevJumpPrior","page":"API","title":"Beluga.BMRevJumpPrior","text":"BMRevJumpPrior\n\nBivariate autocorrelated rates (Brownian motion) prior inspired by coevol (Lartillot & Poujol 2010) with an Inverse Wishart prior on the unknown covariance 2×2 matrix. Crucially, this is defined for the Node based model, i.e. states at model nodes are assumed to be states at nodes of the phylogeny.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.BranchKernel","page":"API","title":"Beluga.BranchKernel","text":"BranchKernel\n\nReversible jump kernel that introduces a WGD, decreases λ and increases μ on the associated branch.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.CRRevJumpPrior","page":"API","title":"Beluga.CRRevJumpPrior","text":"CRRevJumpPrior\n\nConstant-rates model.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.DLWGD","page":"API","title":"Beluga.DLWGD","text":"DLWGD{T<:Real,V<:ModelNode{T}}\n\nDuplication, loss and WGD model. This holds a dictionary for easy access of the nodes in the probabilistic graphical model and the leaf names.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.DLWGD-Tuple{DataFrames.DataFrameRow}","page":"API","title":"Beluga.DLWGD","text":"(m::DLWGD)(row::DataFrameRow)\n\nInstantiate a model based on a row from a trace data frame. This returns a modified copy of the input model.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.DropKernel","page":"API","title":"Beluga.DropKernel","text":"DropKernel\n\nReversible jump kernel that introduces a WGD and decreases λ on the associated branch.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.IRRevJumpPrior","page":"API","title":"Beluga.IRRevJumpPrior","text":"IRRevJumpPrior\n\nBivariate uncorrelated rates prior with an Inverse Wishart prior on the unknown covariance 2×2 matrix. Crucially, this is defined for the Branch based model, i.e. states at model nodes are assumed to be states at branches of the phylogeny.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.PArray","page":"API","title":"Beluga.PArray","text":"PArray{T<:Real}\n\nDitributed array of phylogenetic profiles.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.PostPredSim","page":"API","title":"Beluga.PostPredSim","text":"PostPredSim(chain, data::DataFrame, n::Int64)\n\nPerform posterior predictive simulations.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.Profile","page":"API","title":"Beluga.Profile","text":"Profile{T<:Real}\n\nStruct for a phylogenetic profile of a single family. Geared towards MCMC applications (temporary storage fields) and parallel applications (using DArrays). See also PArray.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.RevJumpChain","page":"API","title":"Beluga.RevJumpChain","text":"RevJumpChain\n\nReversible jump chain struct for DLWGD model inference.\n\n!!! note After construction, an explicit call to init! is required.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.SimpleKernel","page":"API","title":"Beluga.SimpleKernel","text":"SimpleKernel\n\nReversible jump kernel that only introduces a WGD while not chnging λ or μ.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.addwgd!-Union{Tuple{T}, Tuple{DLWGD{T,V} where V<:Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}},Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}},T,T}} where T<:Real","page":"API","title":"Beluga.addwgd!","text":"addwgd!(d::DLWGD, n::ModelNode, t, q)\n\nInsert a WGD node with retention rate q at distance t above node n.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.addwgds!-Tuple{DLWGD,DistributedArrays.DArray{Profile{T},1,Array{Profile{T},1}} where T<:Real,Array}","page":"API","title":"Beluga.addwgds!","text":"addwgds!(m::DLWGD, p::PArray, config::Array)\n\nAdd WGDs from array of named tuples e.g. [(lca=\"ath,cpa\", t=rand(), q=rand())] and update the profile array.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.addwgds!-Tuple{DLWGD,DistributedArrays.DArray{Profile{T},1,Array{Profile{T},1}} where T<:Real,String}","page":"API","title":"Beluga.addwgds!","text":"addwgds!(m::DLWGD, p::PArray, config::String)\naddwgds!(m::DLWGD, p::PArray, config::Dict{Int64,Tuple})\n\nAdd WGDs from a (stringified) dictionary (as in the wgds column of the trace data frame in rjMCMC applications) and update the profile array.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.asvector-Tuple{DLWGD}","page":"API","title":"Beluga.asvector","text":"asvector(d::DLWGD)\n\nGet a parameter vector for the DLWGD model, structured as [λ1, …, λn, μ1, …, μn, q1, …, qk, η].\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.bayesfactors-Tuple{Any}","page":"API","title":"Beluga.bayesfactors","text":"bayesfactors(trace::DataFrame, model::DLWGD, p::Float64)\n\nCompute Bayes Factors for all branch WGD configurations. Returns a data frame that is more or less self-explanatory.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.getrates-Union{Tuple{DLWGD{T,V} where V<:Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}}}, Tuple{T}} where T","page":"API","title":"Beluga.getrates","text":"getrates(model::DLWGD{T})\n\nGet the duplication and loss rate matrix (2 × n).\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.getwgdtrace-Tuple{RevJumpChain}","page":"API","title":"Beluga.getwgdtrace","text":"getwgdtrace(chain)\n\nSummarize all WGD models from an rjMCMC trace. This provdes the data to evaluate retention rates for WGD models etc. Returns a dict of dicts with data frames (which is a horrible data structure, I know) structured as (branch1 => (1 WGD => trace, 2 WGDs => trace, ...), branch2 => (), ...).\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.gradient-Union{Tuple{T}, Tuple{DLWGD,DistributedArrays.DArray{Profile{T},1,Array{Profile{T},1}}}} where T","page":"API","title":"Beluga.gradient","text":"gradient!(d::DLWGD, p::PArray{T})\n\nAccumulate the gradient ∇ℓ(λ,μ,q,η|X) in parallel for the phylogenetic profile matrix p.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.gradient-Union{Tuple{T}, Tuple{DLWGD{T,V} where V<:Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}},Array{Int64,1}}} where T<:Real","page":"API","title":"Beluga.gradient","text":"gradient(d::DLWGD, x::Vector)\n\nCompute the gradient of the log likelihood under the DLWGD model for a single count vector x, ∇ℓ(λ,μ,q,η|x).\n\nwarning: Warning\nCurrently the gradient seems to only work in NaN safe mode github issue\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.init!-Tuple{RevJumpChain}","page":"API","title":"Beluga.init!","text":"init!(chain::RevJumpChain)\n\nInitialize the chain.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.posteriorE!-Tuple{Any}","page":"API","title":"Beluga.posteriorE!","text":"posteriorE!(chain)\n\nCompute E[Xi|Xparent=1,λ,μ] for the joint posterior; i.e. the expected number of lineages at node i under the linear birth-death process given that there was one lineage at the parent of i, for each sample from the posterior. This can give an idea of gene family expansion/contraction.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.posteriorΣ!-Tuple{Any}","page":"API","title":"Beluga.posteriorΣ!","text":"posteriorΣ!(chain)\n\nSample the covariance matrix of the bivariate process post hoc from the posterior under the Inverse Wishart prior. Based on Lartillot & Poujol 2010.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.pppvalues-Tuple{PostPredSim}","page":"API","title":"Beluga.pppvalues","text":"pppvalues(pps::PostPredSim)\n\nCompute posterior predictive p-values based on the posterior predictive distribution and the observed sumary statistics (see e.g. Gelman et al. 2013).\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.removewgd!","page":"API","title":"Beluga.removewgd!","text":"removewgd!(d::DLWGD, n::ModelNode, reindex::Bool=true, set::Bool=true)\n\nRemove WGD/T node n from the DLWGD model. If reindex is true, the model nodes are reindexed to be consecutive. If set is true, the model internals (transition and extinction probabilities) are recomputed.\n\n\n\n\n\n","category":"function"},{"location":"api/#Beluga.removewgds!-Tuple{DLWGD}","page":"API","title":"Beluga.removewgds!","text":"removewgds(d::DLWGD)\n\nRemove all WGD nodes from the model.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.setrates!-Union{Tuple{T}, Tuple{DLWGD{T,V} where V<:Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}},Array{T,2}}} where T","page":"API","title":"Beluga.setrates!","text":"setrates!(model::DLWGD{T}, X::Matrix{T})\n\nSet duplication and loss rates for each non-wgd node|branch in the model. Rates should be provided as a 2 × n matrix, where the columns correspond to model node indices.\n\n\n\n\n\n","category":"method"},{"location":"api/#Distributions.logpdf!-Union{Tuple{T}, Tuple{Array{T,2},DLWGD{T,V} where V<:Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}},Array{Int64,1}}} where T<:Real","page":"API","title":"Distributions.logpdf!","text":"logpdf!(L::Matrix, d::DLWGD, x::Vector{Int64})\n\nCompute the log likelihood under the DLWGD model for a single count vector x ℓ(λ,μ,q,η|x) and update the dynamic programming matrix (L).\n\n\n\n\n\n","category":"method"},{"location":"api/#Distributions.logpdf!-Union{Tuple{T}, Tuple{Array{T,2},Union{NewickTree.TreeNode{Beluga.Branch{T}}, NewickTree.TreeNode{Beluga.Node{T}}},Array{Int64,1}}} where T<:Real","page":"API","title":"Distributions.logpdf!","text":"logpdf!(L::Matrix, n::ModelNode, x::Vector{Int64})\n\nCompute the log likelihood under the DLWGD model for a single count vector x ℓ(λ,μ,q,η|x) and update the dynamic programming matrix (L), only recomputing the matrix above node n.\n\n\n\n\n\n","category":"method"},{"location":"api/#Distributions.logpdf!-Union{Tuple{T}, Tuple{DLWGD,DistributedArrays.DArray{Profile{T},1,Array{Profile{T},1}}}} where T","page":"API","title":"Distributions.logpdf!","text":"logpdf!(d::DLWGD, p::PArray{T})\nlogpdf!(n::ModelNode, p::PArray{T})\n\nAccumulate the log-likelihood ℓ(λ,μ,q,η|X) in parallel for the phylogenetic profile matrix. If the first argument is a ModelNode, this will recompute the dynamic programming matrices starting from that node to save computation. Assumes (of course) that the phylogenetic profiles are iid from the same DLWGD model.\n\n\n\n\n\n","category":"method"},{"location":"api/#Distributions.logpdf-Tuple{DLWGD,Array{Int64,1}}","page":"API","title":"Distributions.logpdf","text":"logpdf(d::DLWGD, x::Vector{Int64})\n\nCompute the log likelihood under the DLWGD model for a single count vector x ℓ(λ,μ,q,η|x).\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.AMMProposals","page":"API","title":"Beluga.AMMProposals","text":"AMMProposals(d)\n\nAdaptive Mixture Metropolis (AMM) proposals, where d is the dimensionality of the rates vector (should be 2 × number of nodes in tree).\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.ConstantDistribution","page":"API","title":"Beluga.ConstantDistribution","text":"ConstantDistribution(x)\n\nA 'constant' distribution (Dirac mass), sometimes useful.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.MWGProposals","page":"API","title":"Beluga.MWGProposals","text":"MWGProposals\n\nProposals for the Metropolis-within-Gibbs algorithm. The MWG algorithm iterates over each node in the tree, resulting in very good mixing and fast convergence in terms of number of iterations, but has quite a high computational cost per generation.\n\n\n\n\n\n","category":"type"},{"location":"api/#Beluga.UpperBoundedGeometric","page":"API","title":"Beluga.UpperBoundedGeometric","text":"UpperBoundedGeometric{T<:Real}\n\nAn upper bounded geometric distribution, basically a constructor for a DiscreteNonParametric distribution with the relevant probabilities.\n\n\n\n\n\n","category":"type"},{"location":"api/#Base.rand-Tuple{DLWGD,Int64}","page":"API","title":"Base.rand","text":"rand(d::DLWGD, N::Int64 [; condition::Vector{Vector{Symbol}}])\n\nSimulate N random phylogenetic profiles under the DLWGD model, subject to the constraint that there is at least one gene in each clade specified in the condition array. (by default conditioning is on non-extinction).\n\nExamples:\n\n```julia-repl julia> # include completely extinct families julia> rand(d, N, condition=[])\n\njulia> # condition on at least one lineage in both clades stemming from the root julia> rand(d, N, condition=Beluga.rootclades(d))\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.rand-Tuple{DLWGD}","page":"API","title":"Base.rand","text":"rand(d::DLWGD)\n\nSimulate a random phylogenetic profile from the DLWGD model.\n\n\n\n\n\n","category":"method"},{"location":"api/#Beluga.set!-Tuple{DLWGD}","page":"API","title":"Beluga.set!","text":"set!(d::DLWGD)\n\nCompute all model internals in postorder.\n\n\n\n\n\n","category":"method"},{"location":"rjmcmc/#Reversible-jump-MCMC-for-the-inference-of-WGDs-in-a-phylogeny-1","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for the inference of WGDs in a phylogeny","text":"","category":"section"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"Load Beluga and required packages:","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"using Beluga, CSV, Distributions","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"Or, if you are running a julia session with multiple processes (when you started julia with -p option, or manually added workers using addprocs, see julia docs), run","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"using CSV, Distributions\n@everywhere using Beluga  # if julia is running in a parallel environment","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"Then get some data (for instance from the example directory of the git repo)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"nw = readline(\"../../example/9dicots/9dicots.nw\")\ndf = CSV.read(\"../../example/9dicots/9dicots-f01-25.csv\")\nmodel, data = DLWGD(nw, df, 1.0, 1.2, 0.9)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"model now refers to the duplication-loss and WGD model (with no WGDs for now), data refers to the phylogenetic profile matrix. The model was initialized with all duplication and loss rates set to 1 and 1.2 respectively. You can check this easily:","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"getrates(model)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"or to get the full parameter vector:","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"asvector(model)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"and you can easily compute likelihoods (and gradients)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"logpdf!(model, data)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"Now we proceed to the hierarchical model. There is no DSL available (à la Turing.jl, Mamba.jl or Soss.jl) but we use a fairly flexible prior struct. Here is an exaple for the (recommended) independent rates (IR) prior:","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"prior = IRRevJumpPrior(\n    # prior covariance matrix (Inverse Wishart prior)\n    Ψ=[1 0. ; 0. 1],\n\n    # multivariate prior on the mean duplication and loss rates\n    X₀=MvNormal([0., 0.], [1 0. ; 0. 1]),  \n\n    # prior on the number of WGDs (can be any discrete distribution)\n    πK=DiscreteUniform(0,20),\n\n    # prior on the WGD retention rate (should be `Beta`)\n    πq=Beta(1,1),\n\n    # prior on η\n    πη=Beta(3,1),\n\n    # tree length (determines prior on WGD times)\n    Tl=treelength(model))","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"We can then construct a chain and run it","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"chain = RevJumpChain(data=data, model=model, prior=prior)\ninit!(chain)\nrjmcmc!(chain, 100, show=10)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"This will log a part of the trace to stdout every show iterations, so that we're able to monitor a bit whether everything looks sensible. Of course in reality you would sample way longer than n=100 iterations.","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"Now we can analyze the trace (in chain.trace), write it to a file, etc. We can also compute Bayes factors to get an idea of the number of WGDs for each branch in the species tree.","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"bayesfactors(chain, burnin=10)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"There are some plotting recipes included in the BelugaPlots package. You may want to try out the following:","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"using BelugaPlots, Plots, StatsPlots\nBelugaPlots.traceplot(chain.trace, burnin=10)\nBelugaPlots.traceplot(chain.trace, burnin=10, st=density)\nBelugaPlots.bfplot(bayesfactors(chain, burnin=10))\nBelugaPlots.ktraceplot(chain.trace, burnin=10)\nposteriorE!(chain)\nBelugaPlots.eplot(chain.trace)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"To obtain a tree with the marginal posterior mean duplication and loss rate estimates, you can try the following (note this is not likely to give you a very nice image unless your taxon IDs are three letter codes)","category":"page"},{"location":"rjmcmc/#","page":"Reversible-jump MCMC for WGD inference","title":"Reversible-jump MCMC for WGD inference","text":"BelugaPlots.doubletree(chain, burnin=10)","category":"page"},{"location":"#Beluga.jl-1","page":"Introduction","title":"Beluga.jl","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"Beluga is a julia library for the statistical analysis of gene family evolution using phylogenetic birth-death processes. It's somewhat related to Whale.jl as it implements models of duplication loss and whole-genome duplication, but employs gene count data instead of gene trees. The library implements the MCMC sampler of Zwaenepoel & Van de Peer (2019) as well as the reversible-jump MCMC sampler of Zwaenepoel & Van de Peer (2020, in preparation).","category":"page"},{"location":"#Data-preparation-1","page":"Introduction","title":"Data preparation","text":"","category":"section"},{"location":"#","page":"Introduction","title":"Introduction","text":"To perform analyses with Beluga, you will need  ","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"An ultrametric species tree (time tree)\nA phylogenetic profile matrix. If you have a bunch of protein fasta files for a set of species of interest, this can be easily obtained using e.g. OrthoFinder.","category":"page"},{"location":"#","page":"Introduction","title":"Introduction","text":"note: Note\nIf you only need the phylogenetic profile matrix from OrthoFinder, be sure to use the -og flag to stop the OrthoFinder pipeline after orthogroup inference. The phylogenetic profile matrix can be found in the Orthogroups.GeneCount.tsv file.","category":"page"},{"location":"mle/#Maximum-likelihood-estimation-1","page":"Maximum-likelihood estimation","title":"Maximum likelihood estimation","text":"","category":"section"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"We'll need the folowing packages loaded","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"using Beluga, CSV, DataFrames, Optim","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"First let's get a random data set","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"nw = \"(D:0.1803,(C:0.1206,(B:0.0706,A:0.0706):0.0499):0.0597);\"\nm, _ = DLWGD(nw, 1.0, 1.5, 0.8)\ndf = rand(m, 1000, condition=Beluga.rootclades(m))\nfirst(df, 5)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"This illustrates how to simulate data from a DLWGD model instance. We created a DLWGD model for the species tree in the Newick string nw with constant rates of duplication (1.0) and loss (1.5) across the tree and a geometric prior distribution on the number of lineages at the root with mean 1.25 (η = 0.8). We then simulated 1000 gene family profiles subject to the condition that there is at least one gene observed at the leaves in each clade stemming from the root.","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"No we set up some stuff to do maximum likelihood estimation for the constant rates model. But first we'll have to get the proper data structure for the profiles","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"model, data = DLWGD(nw, df)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"Note that we already got a model object above, however some internals of the model are dependent on the data to which it will be applied (due to the algorithm of Csuros & Miklos).","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"Now for some functions needed for the ML estimation","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"n = Beluga.ne(model) + 1  # number of nodes = number of edges + 1\nfullvec(θ, η=0.8, n=n) = [repeat([exp(θ[1])], n); repeat([exp(θ[2])], n) ; η]\nf(θ) = -logpdf!(model(fullvec(θ)), data)\ng!(G, θ) = G .= -1 .* Beluga.gradient_cr(model(fullvec(θ)), data)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"The default DLWGD model is parameterized with a rate for each node in the tree, and a DLWGD model with n nodes and k WGDs can be constructed based on a given DLWGD model instance and a vector looking like [[λ1 … λn] ; [μ1 … μn] ; [q1 … qk] ; η]. To be clear, consider the following example code","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"@show v = asvector(model)\nnewmodel = model(rand(length(v)))\n@show asvector(newmodel)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"In the constant rates model, we assume all duplication rates and all loss rates are identical across the tree respectively. The fullvec(θ) function defined above will construct a full model vector, from which we can construct a DLWGD instance, from the simpler vector θ = [log(λ), log(μ)] that defines the constant rates model.","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"Now for optimization. Let's try two optimization algorithms, one only using the likelihood (using the Nelder-Mead downhill simplex algorithm), and another using gradient information (using the LBFGS algorithm)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"init = randn(2)\nresults = optimize(f, init)\n@show exp.(results.minimizer)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"init = randn(2)\nresults = optimize(f, g!, init)\n@show exp.(results.minimizer)","category":"page"},{"location":"mle/#","page":"Maximum-likelihood estimation","title":"Maximum-likelihood estimation","text":"warning: Warning\nCurrently the gradient seems to only work in NaN safe mode. In order to enable NaN safe mode, you should change a line in the ForwardDiff source code. On Linux, and assuming you use julia v1.3, the following should work for most people:rm -r ~/.julia/compiled/v1.3/ForwardDiff\nsed -i 's/NANSAFE_MODE_ENABLED = false/NANSAFE_MODE_ENABLED = true/g' \\\n    ~/.julia/packages/ForwardDiff/*/src/prelude.jl","category":"page"}]
}