using Beluga
using CSV, DataFrames, Parameters

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
# sum(t) / n = 0.02461650895999999
# sum(t) / n = 0.02283291602200001

for i in d.tree.order
    t = []
    n = 50
    for j = 1:n
        d[:λ, i] = rand()
        x = @timed logpdf!(d, p, d.tree.pbranches[i])
        push!(t, x[2])
    end
    @show i, sum(t)/n
end

# (i, sum(t) / n) = (3, 0.008826060279999999)
# (i, sum(t) / n) = (6, 0.015253817519999999)
# (i, sum(t) / n) = (8, 0.01518640452)
# (i, sum(t) / n) = (9, 0.014801436740000002)
# (i, sum(t) / n) = (7, 0.015320238639999994)
# (i, sum(t) / n) = (5, 0.013001249340000005)
# (i, sum(t) / n) = (11, 0.013081707980000003)
# (i, sum(t) / n) = (12, 0.012715384600000004)
# (i, sum(t) / n) = (10, 0.012854052179999997)
# (i, sum(t) / n) = (4, 0.010404697300000001)
# (i, sum(t) / n) = (2, 0.007899255059999997)
# (i, sum(t) / n) = (15, 0.009712675719999998)
# (i, sum(t) / n) = (16, 0.00993919674)
# (i, sum(t) / n) = (14, 0.009796768760000001)
# (i, sum(t) / n) = (18, 0.0096295749)
# (i, sum(t) / n) = (19, 0.009904574299999998)
# (i, sum(t) / n) = (17, 0.01000316818)
# (i, sum(t) / n) = (13, 0.00819302178)
# (i, sum(t) / n) = (1, 0.0006065797799999999)

λ = [0.56, 0.97, 0.12, 0.21, 4.23, 0.12, 0.77, 2.01, 2.99, 0.44]
μ = [0.64, 0.36, 0.25, 0.57, 0.42, 0.49, 0.61, 0.69, 3.92, 0.53]
η = [0.55, 0.56, 0.93, 0.57, 0.59, 0.06, 0.15, 0.41, 0.96, 0.12]
n = length(s)
for i=1:length(l)
    d = DuplicationLoss(s, repeat([λ[i]], n), repeat([μ[i]], n), η[i], m)
    lhood = logpdf!(d, p)
    @show lhood
end

# lhood = -1296.1763022581217
# lhood = -1408.4459849625102
# lhood = -1172.356609929616
# lhood = -1197.7833487345943
# lhood = -3849.887791929266
# lhood = -1348.7468163817248
# lhood = -1486.4818702521393
# lhood = -2083.8730465634676
# lhood = -2351.3561862674096
# lhood = -1364.8837334043392
