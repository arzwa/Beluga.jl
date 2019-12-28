

@with_kw mutable struct TreeLayout
    coords::Dict{Int64,Tuple}
    paths ::Array{Tuple}
    values::Dict{Int64,Float64}    = Dict{Int64,Float64}()
    colors::Dict{Int64,RGB}        = Dict{Int64,RGB}()
    range ::Tuple{Float64,Float64} = (0., 0.)
end

TreeLayout(t::DLWGD, args...) = TreeLayout(t[1], args...)

function TreeLayout(chain; burnin=1000)
    TreeLayout(chain.trace, chain.model; burnin=burnin)
end

function TreeLayout(trace, model; burnin=1000)
    df = trace[burnin+1:end,:]
    lcols = [Symbol("λ$i") for i=1:Beluga.ne(model)+1]
    mcols = [Symbol("μ$i") for i=1:Beluga.ne(model)+1]
    l = mean(log10.(Matrix(df[!,lcols])), dims=1)
    m = mean(log10.(Matrix(df[!,mcols])), dims=1)
    d = deepcopy(model)
    Beluga.setrates!(d, [l ; m])
    mn = minimum([l ; m])
    mx = maximum([l ; m])
    tl1 = TreeLayout(d, :λ)
    tl2 = TreeLayout(d, :μ)
    setcolors!(tl1, mn, mx)
    setcolors!(tl2, mn, mx)
    (tl1, tl2)
end

function TreeLayout(t::TreeNode, s::Symbol=:λ)
    coords = Dict{Int64,Tuple}()
    paths  = Tuple[]
    values = Dict{Int64,Float64}()
    root = t
    yleaf = -1.
    function walk(n)
        values[n.i] = Beluga.isawgd(n) ? Beluga.nonwgdchild(n)[s] : n[s]
        if isleaf(n)
            yleaf += 1
            x = Beluga.parentdist(n, root)
            coords[n.i] = (x, yleaf)
            return yleaf
        else
            u = []
            for c in n.c
                push!(u, walk(c))
                push!(paths, (n.i, c.i))
            end
            y = sum(u)/length(u)
            x = Beluga.parentdist(n, root)
            coords[n.i] =(x, y)
            return y
        end
    end
    walk(root)
    tl = TreeLayout(coords=coords, paths=paths, values=values)
    setcolors!(tl)
    tl
end

function setcolors!(tl, mn=minimum(values(tl.values)),
        mx=maximum(values(tl.values)); cs=ColorSchemes.viridis,)
    for (k, v) in tl.values
        tl.colors[k] = get(cs, (v - mn)/(mx - mn))
    end
    tl.range = (mn, mx)
end

function drawtree!(tl::TreeLayout)
    # quite general, for colored and normal trees
    # linewidth etc. are assumed to be set
    @unpack coords, paths, colors = tl
    for (a,b) in paths
        p1 = Point(coords[a]...)
        p2 = Point(coords[b]...)
        p3 = Point(p1.x, p2.y)
        setblend(blend(p1, p3, colors[a], colors[b]))
        poly([p1, p3, p2])
        Luxor.strokepath()
    end
end

function leaflabels!(tl::TreeLayout, labels)
    for (k, v) in labels
        p = tl.coords[k]
        q = Luxor.Point(0., p[2])
        settext(string(v), q, valign="center")
    end
end

function dimensions(tl::TreeLayout)
    x = [x[1] for (k,x) in tl.coords]
    y = [x[2] for (k,x) in tl.coords]
    minimum(x), maximum(x), minimum(y), maximum(y)
end

function getcolorbar(; breaks=10)
    path = []
    c = 0.
    y = 0.
    Δy = 1/breaks
    for i = 1:breaks
        push!(path, (Luxor.Point(0., y), Luxor.Point(0., y+Δy), 1-y, 1-y+Δy))
        y += Δy
    end
    return path
end

# draw a colorbar
function drawcolorbar!(cbar, minv, maxv; pad=0.2)
    for p in cbar
        ca = get(ColorSchemes.viridis, p[3])
        cb = get(ColorSchemes.viridis, p[4])
        bl = Luxor.blend(p[1], p[2], cb, ca)
        Luxor.line(p[1], p[2])
        Luxor.setblend(bl)
        Luxor.strokepath()
    end
    Luxor.setcolor("black")
    s1 = @sprintf " %3.2f" minv
    s2 = @sprintf " %3.2f" maxv
    p1 = Luxor.Point(cbar[1][1].x   + cbar[1][1].x*pad,   cbar[1][1].y)
    p2 = Luxor.Point(cbar[end][2].x + cbar[end][2].x*pad, cbar[end][2].y)
    Luxor.settext(s2, p1, valign="center", halign="left")
    Luxor.settext(s1, p2, valign="center", halign="left")
end

function internalnodes!(tl::TreeLayout, model::DLWGD)
    for (n,x) in tl.coords
        isleaf(model[n]) ? continue : nothing
        Luxor.settext(" $n", Luxor.Point(x...), valign="center", halign="left")
    end
end



# # using plots; the line_z arg could be used for gradients ____________________
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
