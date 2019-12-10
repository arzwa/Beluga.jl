# MCMCChains interface, diagnostics etc.
function MCMCChains.Chains(trace::DataFrame, burnin=1000)
    df = select(trace, Not([:wgds]))
    for col in names(df)
        df[!,col] .= coalesce.(df[!,col], 0.)
    end
    @assert size(df)[1] > burnin "# MCMC generations < $burnin (burnin)"
    X = reshape(Matrix(df), (size(df)...,1))[burnin+1:end, 2:end, :]
    return Chains(X, [string(x) for x in names(df)][2:end])
end

# get an MCMCChains chain (gives diagnostics etc.)
"""
    Chains(chain::RevJumpChain, burnin)

Obtain and MCMCChains object, with summary and diagnostic statistics.
"""
MCMCChains.Chains(c::RevJumpChain, burnin=1000) = Chains(c.trace, burnin)

"""
    get_wgdtrace(chain)

Summarize all WGD models from an rjMCMC trace. This provdes the data to
evaluate retention rates for WGD models etc. Returns a dict of dicts with
data frames (I know, horrible data structure) which is structured as
(branch_1 => (1 WGD => trace, 2 WGDs => trace, ...), branch_2 => (), ...).
"""
function get_wgdtrace(chain::RevJumpChain)
    tr = Dict{Int64,Dict{Int64,DataFrame}}()
    for (gen, d) in enumerate(chain.trace[!,:wgds])
        for (k,v) in d
            n = length(v)
            if !haskey(tr, k)
                tr[k] = Dict{Int64,DataFrame}()
            end
            if !haskey(tr[k], n)
                tr[k][n] = DataFrame(
                    :gen =>Int64[],
                    [Symbol("q$i")=>Float64[] for i=1:n]...,
                    [Symbol("t$i")=>Float64[] for i=1:n]...)
            end
            sort!(v)
            t = [x[1] for x in v]
            q = [x[2] for x in v]
            push!(tr[k][n], vcat(gen, q, t))
        end
    end
    tr
end


# Bayes factors
# =============
# Prior probabilities for # of WGDs on each branch
function kbranch_prior(k, t, T, prior::UpperBoundedGeometric)
    kmax = prior.b
    f = t/T
    sum([binomial(i, k)*f^k*(1. - f)^(i-k)*pdf(prior, i) for i=k:kmax])
end

# closed form under Geometric prior
function kbranch_prior(k, t, T, prior::Geometric)
    p = prior.p
    a = t/T
    b = 1. - t/T
    q = 1. - p
    p * (a*q)^k * (1. - b*q)^(-k-1)
end

function branch_bayesfactors(chain, burnin::Int64=1000)
    @unpack trace, model, prior = chain
    trace_ = trace[burnin+1:end,:]
    df = branch_bayesfactors(trace_, model, prior.πK)
    show_bayesfactors(df)
    df
end

"""
    branch_bayesfactors(trace::DataFrame, model::DLWGD, p::Float64)

Compute Bayes Factors for all branch WGD configurations. Returns a data frame
that is more or less self-explanatory.
"""
function branch_bayesfactors(trace::DataFrame, model::DLWGD, p)
    T = treelength(model)
    df = DataFrame()
    for (i,n) in sort(model.nodes)
        if isawgd(n); break; end
        if !(Symbol("k$i") in names(trace)); continue; end
        fmap = freqmap(trace[!,Symbol("k$i")])
        kmax = maximum(keys(fmap))
        te = parentdist(n, nonwgdparent(n.p))
        ps = [haskey(fmap, i) ? fmap[i] : 0. for i in 0:kmax]
        πs = kbranch_prior.(0:kmax, te, T, p)
        cl = Beluga.clade(model, n)
        df_ = DataFrame(:branch=>i,
            :clade=>repeat([cl], kmax+1),
            :k=>0:kmax, :eval1=>"", :eval2=>"",
            :p1=>ps, :p0=>[NaN ; ps[1:end-1]],
            :π1=>πs, :π0=>[NaN ; πs[1:end-1]],
            :t=>te, :tfrac=>te/T)
        df = vcat(df, df_)
    end
    df[!,:p2] = by(df, :branch, :p1=>cumsum0)[!,:p1_cumsum0]
    df[!,:π2] = by(df, :branch, :π1=>cumsum0)[!,:π1_cumsum0]
    # compute BFs
    df[!,:K1] = (df[!,:p1] ./ df[!,:p0]) ./ (df[!,:π1] ./ df[!,:π0])
    df[!,:K2] = (df[!,:p2] ./ df[!,:p0]) ./ (df[!,:π2] ./ df[!,:π0])
    df[!,:log10K1] = log10.(df[!,:K1])
    df[!,:log10K2] = log10.(df[!,:K2])
    for x in [0.5, 1., 2.]
        df[!,:eval1][df[!,:log10K1] .> x] .*= "*"
        df[!,:eval2][df[!,:log10K2] .> x] .*= "*"
    end
    df
end

cumsum0(x) = 1 .- vcat(0., cumsum(x)...)[1:end-1]

function show_bayesfactors(df::DataFrame)
    for d in groupby(df, :branch)
        @printf "🌲 %2d: " d[1,:branch]
        println("(", join(string.(d[1,:clade]), ","), ")")
        for (i,r) in enumerate(eachrow(d))
            isnan(r[:p0]) ? continue : nothing
            @printf "[%2d vs. %2d] " i-1 i-2
            @printf "K = (%.2f/%.2f) ÷ (%.2f/%.2f) " r[:p1] r[:p0] r[:π1] r[:π0]
            @printf "= %8.3f [log₁₀(K) = %8.3f] %s\n" r[:K1] r[:log10K1] r[:eval1]
        end
        println()
        for (i,r) in enumerate(eachrow(d))
            isnan(r[:p0]) ? continue : nothing
            @printf "[>%1d vs. %2d] " i-1 i-2
            @printf "K = (%.2f/%.2f) ÷ (%.2f/%.2f) " r[:p2] r[:p0] r[:π2] r[:π0]
            @printf "= %8.3f [log₁₀(K) = %8.3f] %s\n" r[:K2] r[:log10K2] r[:eval2]
        end
        println("_"^78)
    end
end

# posterior expectations
function posterior_E!(chain)
    @unpack trace, model = chain
    for i=2:ne(model)+1
        t = parentdist(model[i], nonwgdparent(model[i].p))
        l = trace[!,Symbol("λ$i")]
        m = trace[!,Symbol("μ$i")]
        E = exp.(t*(l .- m))
        V = E .* (E .- 1.) .* (l .+ m) ./ (l .- m)
        trace[!,Symbol("E$i")] .= E
        trace[!,Symbol("V$i")] .= V
    end
end

# see Lartillot & Poujol 2010
function posterior_Σ!(chain)
    @unpack model, prior = chain
    @unpack Σ₀ = prior
    chain.trace[!,:var] .= NaN
    chain.trace[!,:cov] .= NaN
    for row in eachrow(chain.trace)
        m = model(row)
        @unpack A, q, n = scattermat(model, prior)
        Σ = rand(InverseWishart(q + n, Σ₀ + A))
        row[:var] = Σ[1,1]
        row[:cov] = Σ[1,2]
    end
end


# posterior predictive checks/simulations/tests
# =============================================
mutable struct PostPredSim
    datastats::DataFrame
    ppstats  ::DataFrame
end

PostPredSim(chain::RevJumpChain, data, n; kwargs...) =
    PostPredSim(chain.trace, chain.model, data, n; kwargs...)

function PostPredSim(trace::DataFrame, model::DLWGD, data::DataFrame, n::Int64;
        burnin=1000, lstats=[mean, std, entr], gstats=lstats)
    N = size(data)[1]
    ngen = size(trace)[1]
    @assert burnin < ngen "burnin ($burnin) < generations ($ngen)"
    d1 = pp_simulate(trace, model, N, n,
        burnin=burnin, lstats=lstats, gstats=gstats)
    d2 = sstats(data, lstats=lstats, gstats=gstats)
    PostPredSim(d2, d1)
end

function pp_simulate(trace::DataFrame, model::DLWGD, N::Int64, n::Int64;
        burnin=1000, lstats=[mean, std, entr], gstats=lstats)
    clades = [clade(model, x) for x in model[1].c]
    trace = trace[burnin+1:end,:]
    # NOTE: could be in parallel!
    sdf = [sstats(rand(model(rand(trace)), N),
        lstats=lstats, gstats=gstats) for i=1:n]
    vcat(sdf...)
end

function sstats(df::DataFrame;lstats=[mean, std, entr], gstats=lstats)
    ldf = describe(df, [Pair(x...) for x in zip(Symbol.(lstats), lstats)]...)
    gdf = flatten(ldf, :variable)
    for s in gstats
        gdf[!,Symbol(s)] .= s(vcat(Matrix(df)...))
    end
    gdf
end

entr(x) = entropy(values(freqmap(x)))

function flatten(df::DataFrame, on::Symbol)
    d = Dict()
    for r in eachrow(df)
        for n in names(r)
            n != on ? d[Symbol("$(r[on])_$n")] = r[n] : continue
        end
    end
    DataFrame(d)
end

function pp_pvalues(pps::PostPredSim)
    @unpack datastats, ppstats = pps
    nrep = size(ppstats)[1]
    d = Dict{Symbol,Float64}()
    for n in names(ppstats)
        x = datastats[1,n]
        p = length(ppstats[ppstats[!,n] .> x,n])/nrep
        d[n] = p > 0.5 ? 1. - p : p
    end
    df = stack(DataFrame(d))
    df[!,:eval] .= ""
    df[df[!,:value] .< 0.05 ,:eval] .= "*"
    df[df[!,:value] .< 0.01 ,:eval] .= "**"
    df[df[!,:value] .< 0.001,:eval] .= "***"
    df
end
