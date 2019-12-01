using Test, DataFrames, CSV, Distributions, Random
using Beluga

Random.seed!(333)

# setup
datadir = "data"
df1 = DataFrame(:A=>[2,3],:B=>[2,5],:C=>[3,0],:D=>[4,1])
df2 = DataFrame(:A=>rand(1:20, 1000), :B=>rand(1:20, 1000),
    :C=>rand(1:10, 1000), :D=>rand(1:10, 1000))
df3 = CSV.read(joinpath(datadir, "plants1-100.tsv"), delim=",")
s1 = s2 = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
s3 = open(joinpath(datadir, "plants1.nw"), "r") do f ; readline(f); end
s4 = open(joinpath(datadir, "plants1c.nw"), "r") do f ; readline(f); end
shouldbe1 = [-Inf, -13.0321, -10.2906, -8.96844, -8.41311, -8.38041,
    -8.78481, -9.5921, -10.8016, -12.4482, -14.6268, -17.607]
shouldbe2 = [-Inf, -12.6372, -10.112, -8.97727, -8.59388, -8.72674,
    -9.29184, -10.2568, -11.6216, -13.4219, -15.753, -18.8843]
shouldbe3 = [-1296.1763022581217, -1408.4459849625102, -1172.356609929616,
    -1197.7833487345943, -3849.887791929266, -1348.7468163817248,
    -1486.4818702521393, -2083.8730465634676, -2351.3561862674096,
    -1364.8837334043392]

include("tests.jl")
