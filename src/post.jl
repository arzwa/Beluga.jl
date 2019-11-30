# MCMCChains interface, diagnostics etc.
function MCMCChains.Chains(trace::DataFrame, burnin=1000)
    df = deletecols(trace, :wgds)
    if size(df)[1] < burnin
        @error "Trace not long enough to discard $burnin iterations as burn-in"
    end
    X = reshape(Matrix(df), (size(df)...,1))[burnin+1:end, 2:end, :]
    return Chains(X, [string(x) for x in names(df)][2:end])
end

# get an MCMCChains chain (gives diagnostics etc.)
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
    for d in chain.trace[!,:wgds]
        for (k,v) in d
            n = length(v)
            if !haskey(tr, k)
                tr[k] = Dict{Int64,DataFrame}()
            end
            if !haskey(tr[k], n)
                tr[k][n] = DataFrame(
                    [Symbol("q$i")=>Float64[] for i=1:n]...,
                    [Symbol("t$i")=>Float64[] for i=1:n]...)
            end
            sort!(v)
            t = [x[1] for x in v]
            q = [x[2] for x in v]
            push!(tr[k][n], [q ; t])
        end
    end
    tr
end


# Bayes factors
# =============
# Prior probabilities for # of WGDs on each branch
# function kbranch_prior(n, k, T, prior, kmax=100)
#     # NOTE approximate: kmax is the truncation bound for the # of WGDs in total
#     # k is the maximal number of WGDs on the branch of interest for which the
#     # prior should be computed
#     te = parentdist(n, nonwgdparent(n.p))
#     p = Dict(j=>0. for j=0:k)
#     for j=0:k
#         q = (te/T)^j
#         for i=0:kmax
#             p[j] += binomial(i, j)*q*(1. - (te/T))^(i-j)*pdf(prior, i)
#         end
#     end
#     p
# end

# closed form
function kbranch_prior(k, t, T, p)
    a = t/T
    b = 1. - t/T
    q = 1. - p
    p * (a*q)^k * (1. - b*q)^(-k-1)
end

function branch_bayesfactors(chain, burnin::Int64=1000)
    @unpack trace, model, prior = chain
    trace_ = trace[burnin+1:end,:]
    df = branch_bayesfactors(trace_, model, prior.πK.p)
    show_bayesfactors(df)
    df
end

"""
    branch_bayesfactors(trace::DataFrame, model::DLWGD, p::Float64)

Compute Bayes Factors for all branch WGD configurations. Returns a data frame
that is more or less self-explanatory. 
"""
function branch_bayesfactors(trace::DataFrame, model::DLWGD, p::Float64)
    T = treelength(model)
    df = DataFrame()
    for (i,n) in sort(model.nodes)
        if isawgd(n); break; end
        if !(Symbol("k$i") in names(trace)); continue; end
        fmap = freqmap(trace[Symbol("k$i")])
        kmax = maximum(keys(fmap))
        te = parentdist(n, nonwgdparent(n.p))
        ps = [haskey(fmap, i) ? fmap[i] : 0. for i in 0:kmax]
        πs = kbranch_prior.(0:kmax, te, T, p)
        cl = Beluga.clade(model, n)
        df_ = DataFrame(:branch=>i,
            :clade=>repeat([cl], kmax+1),
            :k=>0:kmax, :eval=>"",
            :p1=>ps, :p0=>[NaN ; ps[1:end-1]],
            :π1=>πs, :π0=>[NaN ; πs[1:end-1]],
            :t=>te, :tfrac=>te/T)
        df = vcat(df, df_)
    end
    # compute BFs
    df[:K] = (df[:p1] ./ df[:p0]) ./ (df[:π1] ./ df[:π0])
    df[:log10K] = log10.(df[:K])
    for x in [0.5, 1., 2.]
        df[:eval][df[:log10K] .> x] .*= "*"
    end
    df
end

function show_bayesfactors(df::DataFrame)
    for d in groupby(df, :branch)
        @printf "🌲 %2d: " d[:branch][1]
        println("(", join(string.(d[:clade][1]), ","), ")")
        for (i,r) in enumerate(eachrow(d))
            isnan(r[:p0]) ? continue : nothing
            @printf "[%2d vs. %2d] " i-1 i-2
            @printf "K = (%.2f/%.2f) ÷ (%.2f/%.2f) " r[:p1] r[:p0] r[:π1] r[:π0]
            @printf "= %8.3f [log₁₀(K) = %8.3f] %s\n" r[:K] r[:log10K] r[:eval]
        end
        println("_"^78)
    end
end
