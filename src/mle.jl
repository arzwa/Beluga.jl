# MLE using the Optim library
"""
    mle(d::DLModel, X::PArray, optimizer=LBFGS(); kwargs...)
"""
function mle(d::DuplicationLossWGD, X::AbstractArray, optimizer=LBFGS();
        kwargs...)
    kwargs = merge(Dict(:show_every=>10, :show_trace=>true), kwargs)
    x = asvector(d)
    η = pop!(x)
    f  = (v) -> -logpdf(d(v, η), X)
    g! = (G, v) -> G .= -gradient(d(v, η), X)[1:end-1]
    lower, upper = bounds(d)
    opts = Optim.Options(;kwargs...)
    out = do_opt(optimizer, opts, f, g!, lower, upper, x)
    m = d(out.minimizer, η)
    return m, out
end

function do_opt(optimizer::Optim.FirstOrderOptimizer, opts, args...)
    optimize(args..., Fminbox(optimizer), opts)
end

do_opt(optimizer::Optim.ZerothOrderOptimizer, opts, args...) =
    optimize(args[1], args[3:end]..., Fminbox(optimizer), opts)

# get bounds for DLWGD model
function bounds(d::DuplicationLossWGD)
    n = 2*nrates(d.tree)
    q = nwgd(d.tree)
    lower = [0. for i=1:n+q]
    upper = [[Inf for i=1:n] ; [1 for i=1:q]]
    return lower, upper
end

Base.length(d::DuplicationLossWGD) = length(asvector(d))
