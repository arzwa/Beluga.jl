using Luxor


function coordspaths(d::DLWGD)
    yleaf = -1.
    paths = []
    coords = Dict{Int64,Point}()
    # postorder traversal
    function walk(node)
        if isleaf(node)
            yleaf += 1.
            x = 0.
            coords[node] = Point(x, yleaf)
            return yleaf
        else
            ychildren = []
            x = 0.
            for child in node.c
                push!(ychildren, walk(child))
                push!(paths, (node.i, child.i))
                x = child[:t] + coords[child.i].x
            end
            y = sum(ychildren) / length(ychildren)
            coords[node.i] = Point(x, y)
            return y
        end
    end
    walk(root)
    return coords, paths
end
