using Beluga
using CSV, DataFrames

s = SpeciesTree("test/data/plants1.nw")
df = CSV.read("test/data/plants1-100.tsv", delim=",")
p, m = Profile(df, s)
λ = repeat([0.2], length(s))
μ = repeat([0.3], length(s))
d = DuplicationLoss(s, λ, μ, 0.8, m)

t = []
n = 500
for i=1:n
    x = @timed logpdf!(d, p)
    push!(t, x[2])
end
@show sum(t)/n
# sum(t) / n = 0.024616508959999988
