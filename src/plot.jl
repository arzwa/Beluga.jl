#
# @with_kw struct TreeLayout
#     coords::Dict{Int64,Tuple}
#     values::Dict{Int64,Float64}  = Dict{Int64,Float64}()
#     paths ::Array{Tuple}
# end
#
# TreeLayout(t::DLWGD) = TreeLayout(t[1])
#
# function TreeLayout(t::TreeNode)
#     coords = Dict{Int64,Tuple}()
#     paths  = Tuple[]
#     root = t[1]
#     yleaf = -1.
#
#     function walk(n)
#         if isleaf(n)
#             yleaf += 1
#             x = parentdist(n, root)
#             coords[n.i] = (x, yleaf)
#             return yleaf
#         else
#             u = []
#             for c in n.c
#                 push!(u, walk(c))
#                 push!(paths, (n.i, c.i))
#             end
#             y = sum(u)/length(u)
#             x = parentdist(n, root)
#             coords[n.i] =(x, y)
#             return y
#         end
#     end
#     walk(root)
#     TreeLayout(coords=coords, paths=paths)
# end
#
# # using plots; the line_z arg could be used for gradients
# function tplot(tl::TreeLayout)
#     @unpack paths, coords = tl
#     p = Plots.plot(legend=false, grid=false, yticks=false)
#     for (a, b) in paths
#         p1, p2 = coords[a], coords[b]
#         Plots.plot!([p1[1],p2[1]], [p2[2],p2[2]],
#             color=:black, linewidth=2)
#         Plots.plot!([p1[1],p1[1]], [p1[2],p2[2]],
#             color=:black, linewidth=2)
#     end
#     p
# end
#
# function leafnames!(p, tl, leaves)
#     @unpack coords = tl
#     for (k,v) in leaves
#         annotate!(p, coords[k]..., text("\t$v", :left))
#     end
#     p
# end
#
# function wgds!(p, tl, t)
#     @unpack coords = tl
#     X = zeros(2,0)
#     for n in Beluga.getwgds(t)
#         X = hcat(X, [tl.coords[n]...])
#     end
#     scatter!(p, X[1,:],X[2,:],
#         marker=:rect, markersize=5, color=:white)
# end
#
# function tplot(d::DLWGD)
#     tl = TreeLayout(d)
#     p = tplot(tl)
#     leafnames!(p, tl, d.leaves)
#     wgds!(p, tl, d)
#     p
# end

# Posterior predictive histograms plot recipe
@recipe function f(p::Beluga.PostPredSim;
        vlinecolor=:salmon, vlinewidth=2)
    legend --> false
    xticks --> false
    yticks --> false
    grid   --> false
    n = length(names(p.ppstats))
    layout := n
    for (i,n) in enumerate(sort(names(p.ppstats)))
        @series begin
            linewidth --> 1
            color --> :black
            linecolor --> :black
            seriestype := histogram
            subplot := i
            p.ppstats[!,n]
        end

        @series begin
            title := join(split(string(n), "_"), " ")
            title_loc := :left
            titlefont := font(8)
            foreground := :white
            linewidth := vlinewidth
            color := vlinecolor
            seriestype := vline
            subplot := i
            [p.datastats[1,n]]
        end
    end
end

@userplot TracePlot

@recipe function f(x::TracePlot;
        ncol=5, thin=1, burnin=1000, colors = [:black, :salmon])
    trace = x.args[1]
    trace[!,:logP] = trace[!,:logπ] .+ trace[!,:logp]
    @assert typeof(trace) == DataFrame "Input should be a data frame!"
    trace = trace[burnin:thin:end,:]
    lcols = [col for col in names(trace) if startswith(string(col), "λ")]
    mcols = [col for col in names(trace) if startswith(string(col), "μ")]
    n = length(lcols) + 3
    legend := false
    xticks --> false
    yticks --> false
    grid   := false
    nrow = n % ncol == 0 ? n ÷ ncol : (n ÷ ncol) + 1
    layout := (nrow, ncol)
    for (i, set) in enumerate([lcols, mcols])
        for (j, col) in enumerate(set)
            @series begin
                linewidth  --> 0.5
                alpha      --> 0.8
                color      --> colors[i]
                seriestype --> :path
                subplot     := j
                title       := "\\theta_{$j}"
                title_loc   := :left
                titlefont  --> font(8)
                log.(trace[!,col])
            end
        end
    end
    j = length(lcols)
    for (col, lab) in zip([:k, :η1, :logP], ["k", "\\eta", "logP\\(\\theta|X\\)"])
        j += 1
        @series begin
            linewidth  --> 0.5
            alpha      --> 0.8
            color      --> :black
            seriestype --> :path
            subplot     := j
            title       := lab
            title_loc   := :left
            titlefont  --> font(8)
            foreground --> :auto
            trace[!,col]
        end
    end
end
