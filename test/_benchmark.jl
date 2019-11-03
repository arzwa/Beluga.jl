using Beluga
using DataFrames, CSV

s = open("test/data/plants1.nw", "r") do f
    readline(f)
end

df = CSV.read("test/data/plants1-100.tsv", delim=",")
d, y = DuplicationLossWGDModel(s, df, 0.2, 0.3, 0.8)
p = Profile(y)

t = []
n = 500
for i=1:n
    x = @timed logpdf!(d, p)
    push!(t, x[2])
end
@show sum(t)/n
# sum(t) / n = 0.018538160911999987
