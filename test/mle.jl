using Beluga
using CSV
using DataFrames
using Optim

# get a random data set
nw = "(D:0.1803,(C:0.1206,(B:0.0706,A:0.0706):0.0499):0.0597);"
m, _ = DLWGD(nw, 1.0, 1.5, 0.8)
df = rand(m, 1000, condition=Beluga.rootclades(m))

model, data = DLWGD(nw, df, rand(2)..., 0.9)
n = Beluga.ne(model) + 1

fullvec(θ, η=0.8, n=n) = [repeat([exp(θ[1])], n); repeat([exp(θ[2])], n) ; η]
f(θ) = -logpdf!(model(fullvec(θ)), data)
g!(G, θ) = G .= -1 .* Beluga.gradient_cr(model(fullvec(θ)), data)

init = randn(2)
results = optimize(f,  init)
@show exp.(results.minimizer)

init = randn(2)
results = optimize(f, g!, init)
@show exp.(results.minimizer)
