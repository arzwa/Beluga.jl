# finish this implementation; but maybe think about more elegant implementation
# perhaps using the DAG more abstractly or so? The DLModel could be fully
# represented as a DAG with speciation, WGD and root nodes associated with
# their rates

# at most two objects should change during the rj MCMC: the model (holding the
# tree, ϵ and W) and the PArray, with the DP matrices (add/remove dimensions)
# these methods should presumably work on copies

# add WGD node and modify profile array
# add wgd onm branch e in model m and profile array p at time t with ret. rate q
function Beluga.addwgd!(m, p, e, t, q)
    n, n_ = addwgd!(m.tree, e, t, nwgd(m.tree)+1)
    extend!(m.value)
    m.q = [m.q ; 0.]
    m[:q, n] = q      # this will reocompute the necessary stuff
    extend!(p, e, 2)
end

# remove WGD node and modify profile array
# remove wgd node n
function removewgd!(m, p, n)
    # delete branches
    # delete nodes
    # connect disconnected nodes
    # remove node from bindex
    shrink!(p, [n, n+1])  # modify profile
end

# change WGD time
# tbefore is the time between WGD node n and its parentnode (which may be a WGD)
function shiftwgd!(m, p, n, tbefore)
end


# Change models
function extend!(v::CsurosMiklos{T}) where T<:Real
    v.ϵ = [v.ϵ ; Beluga.minfs(T, 2, 2)]
    v.W = cat(v.W, Beluga.minfs(T, v.m+1, v.m+1, 2), dims=3)
end

function shrink!(v::CsurosMiklos{T}, i::Int64) where T<:Real
    v.ϵ = [v.ϵ[1:i-1,:] ; v.ϵ[i+1:end,:]]
    v.W = cat(v.W[:,:,1:i-1],v.W[:,:,i+1:end], dims=3)
end



# Change profiles
# to profile eventually
# only change Ltmp; L should be set using set_L, something similar for x?
extend!(p::AbstractArray{P}, e::Int64, dim::Int64) where P<:AbstractProfile =
    map((x)->extend!(x, e, dim), p)

function extend!(p::Profile{T}, e::Int64, dim::Int64) where T<:Real
    m = size(p.L)[2]
    p.x = [p.x ; [p.x[e], p.x[e]]]  # profile itself should be extended as well!
    p.Ltmp = [p.Ltmp ; Beluga.minfs(T, 2, m)]
end

function extend!(p::MixtureProfile{T}, e::Int64, dim::Int64) where T<:Real
    m = size(p.L)[2]
    p.x = [p.x ; [p.x[e], p.x[e]]]  # profile itself should be extended as well!
    p.Ltmp = [p.Ltmp ; Beluga.minfs(T, 2, m)]
end

shrink!(p::AbstractArray{P}, i) where P<:AbstractProfile =
    map((x)->shrink!(x, i), p)

function shrink!(p::Profile{T}, i) where T<:Real
    p.x    = [p.x[1:i-1] ; p.x[i+1:]]
    p.L    = [p.L[1:i-1,:] ;    p.L[i+1:end,:]]
    p.Ltmp = [p.Ltmp[1:i-1,:] ; p.Ltmp[i+1:end,:]]
end

function shrink!(p::MixtureProfile{T}, i) where T<:Real
    p.x    = [p.x[1:i-1] ; p.x[i+1:]]
    p.L    = [p.L[1:i-1,:] ;    p.L[i+1:end,:]]
    p.Ltmp = [p.Ltmp[1:i-1,:] ; p.Ltmp[i+1:end,:]]
end
