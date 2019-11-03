# NOTE:
# * update! will be faster than using a deepcopy of the model

function moverates!(d::DLWGD, p::PArray, state::Dict)
    m = length(d)
    x = zeros(2m)
    for n in postwalk(d[1])
        lλ = log(n[:λ])
        lμ = log(n[:μ])
        ll = lλ + randn()/100
        lm = lμ + randn()/100
        update!(n, (λ=exp(ll), μ=exp(lm)))
        l_ = logpdf!(n, p)
        hr = l_ - state[:logp]
        if log(rand()) < hr
            state[:logp] = l_
            set!(p)
        else
            update!(n, (λ=exp(lλ), μ=exp(lμ)))
            rev!(p)
        end
        x[n.i] = n[:λ]
        x[n.i+m] = n[:μ]
    end
    x
end


function kissmcmc!(d, p, n)
    state = Dict(:logp=>logpdf!(d,p))
    for i=1:n
        x = moverates!(d, p, state)
        @show x[1:7], state[:logp]
    end
end
