
s = SpeciesTree("test/data/plants1.nw")
df = CSV.read("test/data/plants1-10.tsv", delim=",")
deletecols!(df, :Orthogroup)

p, m = Profile(df, s)
Beluga.set_constantrates!(s)
d = DuplicationLoss(s, [0.2], [0.2], 0.8, m)

addwgd!(d, p, 18, 0.2, 0.3)
