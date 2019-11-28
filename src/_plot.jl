
struct TreeLayout
    coords::Dict{Int64,Tuple}
    paths ::Array{Tuple}
end

function TreeLayout(t::DLWGD)
    coords = Dict{Int64,Tuple}()
    paths  = Tuple[]
    root = t[1]
    yleaf = -1.

    function walk(n)
        if isleaf(n)
            yleaf += 1
            x = parentdist(n, root)
            coords[n.i] = (x, yleaf)
            return yleaf
        else
            u = []
            for c in n.c
                push!(u, walk(c))
                push!(paths, (n.i, c.i))
            end
            y = sum(u)/length(u)
            x = parentdist(n, root)
            coords[n.i] =(x, y)
            return y
        end
    end
    walk(root)
    TreeLayout(coords, paths)
end

# using luxor
# import Luxor
# import Luxor: Point
function drawtree(tl::TreeLayout)
    @unpack paths, coords = tl
    for (a, b) in paths
        p, q = coords[a], coords[b]
        path = [Point(p...), Point(p[1], q[2]), Point(q...)]
        # setblend
        Luxor.poly(path)
        Luxor.strokepath()
    end
end

# using plots
# using Plots
function tplot(tl::TreeLayout)
    @unpack paths, coords = tl
    p = Plots.plot(legend=false, grid=false, yticks=false)
    for (a, b) in paths
        p1, p2 = coords[a], coords[b]
        Plots.plot!([p1[1],p2[1]], [p2[2],p2[2]],
            color=:black, linewidth=2)
        Plots.plot!([p1[1],p1[1]], [p1[2],p2[2]],
            color=:black, linewidth=2)
    end
    p
end

function leafnames!(p, tl, leaves)
    @unpack coords = tl
    for (k,v) in leaves
        annotate!(p, coords[k]..., text("\t$v", :left))
    end
    p
end

function wgds!(p, tl, t)
    @unpack coords = tl
    X = zeros(2,0)
    for n in Beluga.getwgds(t)
        X = hcat(X, [tl.coords[n]...])
    end
    scatter!(p, X[1,:],X[2,:],
        marker=:rect, markersize=5, color=:white)
end

function tplot(d::DLWGD)
    tl = TreeLayout(d)
    p = tplot(tl)
    leafnames!(p, tl, d.leaves)
    wgds!(p, tl, d)
    p
end
