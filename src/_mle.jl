using Optim


ddir = "test/data"
nw = open(joinpath(ddir, "plants1c.nw"), "r") do f ; readline(f); end
df = CSV.read(joinpath(ddir, "N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv"))
@unpack model, data = DLWGD(nw, df, 1., 1., 0.9)


"""
    mle(m::DLWGD, data::PArray, optimizer=LBFGS(); kwargs...)
"""
function mle(m::DLWGD, data, optimizer=LBFGS(); kwargs...)
    kwargs = merge(Dict(:show_every=>10, :show_trace=>true), kwargs)
    x = asvector(m)
    f  = (x) -> -logpdf!(m(x), data)
    g! = (G, x) -> G .= -gradient(m(x), data)
    lower, upper = bounds(m)
    opts = Optim.Options(;kwargs...)
    out = do_opt(optimizer, opts, f, g!, lower, upper, x)
    return out
end

function crmle(m::DLWGD, data, optimizer=LBFGS(); kwargs...)
    kwargs = merge(Dict(:show_every=>10, :show_trace=>true), kwargs)
    x = Beluga.getrates(m)[:,1]
    n = Beluga.ne(m) + 1
    f  = (x) -> -logpdf!(m(x[1], x[2]), data)
    g! = (G, x) -> G .= -gradient(m(x[1], x[2]), data)
    lower, upper = [0., 0.], [Inf, Inf]
    opts = Optim.Options(;kwargs...)
    out = do_opt(optimizer, opts, f, g!, lower, upper, x)
    return out
end

function (m::DLWGD)(la, mu)
    Beluga.setrates!(m, permutedims(
        [repeat([la], Beluga.ne(m) + 1) repeat([mu], Beluga.ne(m) + 1)]))
    Beluga.set!(m)
    m
end

do_opt(optimizer::Optim.FirstOrderOptimizer, opts, args...) =
    optimize(args..., Fminbox(optimizer), opts)

do_opt(optimizer::Optim.ZerothOrderOptimizer, opts, args...) =
    optimize(args[1], args[3:end]..., Fminbox(optimizer), opts)

function bounds(m::DLWGD)
    lower = [0. for i=1:length(asvector(m))]
    upper = [[Inf for i=1:2*(Beluga.ne(m)+1)] ; [1. for i=1:Beluga.nwgd(m)] ; 1.]
    return lower, upper
end
