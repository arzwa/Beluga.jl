# MLE using the Optim library
"""
    mle(d::DLModel, M::AbstractArray, optimizer=LBFGS(); kwargs...)
"""
function mle(d::DLModel, M::AbstractArray, optimizer=LBFGS(); kwargs...)
    kwargs = merge(Dict(:show_every=>10, :show_trace=>true), kwargs)
    x = asvector(d)
    f  = (v) -> -logpdf(DLModel(d, v), M)
    g! = (G, v) -> G .= -gradient(DLModel(d, v), M)
    lower, upper = bounds(d)
    opts = Optim.Options(;kwargs...)
    out = do_opt(optimizer, opts, f, g!, lower, upper, x)
    m = DLModel(d, out.minimizer)
    return m, out
end

function do_opt(optimizer::Optim.FirstOrderOptimizer, opts, args...)
    optimize(args..., Fminbox(optimizer), opts)
end

do_opt(optimizer::Optim.ZerothOrderOptimizer, opts, args...) =
    optimize(args[1], args[3:end]..., Fminbox(optimizer), opts)

# get bounds for Whale model
function bounds(d::DLModel)
    lower = [0. for i=1:length(asvector(d))]
    upper = [Inf for i=1:length(asvector(d))]
    return lower, upper
end

#=function f(x::Vector)  # using KissThreading
    w = WhaleModel(t, x)
    v0 = -logpdf(w, ccd[1])
    return @views tmapreduce(+, ccd[2:end], init=v0) do c
        -logpdf(w, c)
    end
end=#
