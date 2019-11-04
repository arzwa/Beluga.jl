using Beluga
using DataFrames, CSV, PhyloTree

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
# sum(t) / n = 0.019814680058000002
# sum(t) / n = 0.019345876993999995

for n in postwalk(d[1])
    t = []
    N = 50
    for j = 1:N
        update!(n, :λ, rand())
        x = @timed logpdf!(n, p)
        push!(t, x[2])
    end
    @show n.i, sum(t)/N
end

n = d[1]
t = []
N = 50
for j=1:N
    update!(n, :η, rand())
    x = @timed logpdfroot(n, p)
    push!(t, x[2])
end
@show n.i, sum(t)/N

# (n.i, sum(t) / N) = (3, 0.00790371426)
# (n.i, sum(t) / N) = (6, 0.010732028280000002)
# (n.i, sum(t) / N) = (8, 0.012145728379999998)
# (n.i, sum(t) / N) = (9, 0.01174503884)
# (n.i, sum(t) / N) = (7, 0.011997942159999999)
# (n.i, sum(t) / N) = (5, 0.010341694559999998)
# (n.i, sum(t) / N) = (11, 0.010303690960000002)
# (n.i, sum(t) / N) = (12, 0.010322891679999995)
# (n.i, sum(t) / N) = (10, 0.01040603804)
# (n.i, sum(t) / N) = (4, 0.00844974756)
# (n.i, sum(t) / N) = (2, 0.006786585399999997)
# (n.i, sum(t) / N) = (15, 0.007891426799999999)
# (n.i, sum(t) / N) = (16, 0.0077183322399999995)
# (n.i, sum(t) / N) = (14, 0.007733749240000001)
# (n.i, sum(t) / N) = (18, 0.007835988880000002)
# (n.i, sum(t) / N) = (19, 0.0077634954400000004)
# (n.i, sum(t) / N) = (17, 0.007992182879999999)
# (n.i, sum(t) / N) = (13, 0.006279051480000001)
# (n.i, sum(t) / N) = (1, 0.004396652000000001)   [λ or μ changed at root]
# (n.i, sum(t) / N) = (1, 0.0006618277400000001)  [only η changed at root]
