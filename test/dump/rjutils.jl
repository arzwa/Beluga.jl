using Beluga, DataFrames, CSV, PhyloTrees, Parameters
import Beluga.CsurosMiklos

s = SpeciesTree("test/data/plants1.nw")
df = CSV.read("test/data/plants1-10.tsv", delim=",")
deletecols!(df, :Orthogroup)

p, m = Profile(df, s)
#Beluga.set_constantrates!(s)
d = DuplicationLoss(s, rand(19), rand(19), 0.8, m)

@time addwgd!(d, p, 18, 0.2, 0.2)
@show logpdf!(d, p)
@time addwgd!(d, p, 18, 0.6, 0.3)
@show logpdf!(d, p)
@time removewgd!(d, p, 21)
@show logpdf!(d, p)
@time addwgd!(d, p, 8, 0.6, 0.3)
@time addwgd!(d, p, 12, 0.6, 0.3)
@time addwgd!(d, p, 14, 0.9, 0.3)
@show logpdf!(d, p)

@show parentdist(s, 14)
shiftwgd!(d, 14, 0.2)
@show parentdist(s, 14)
@show logpdf!(d, p)
