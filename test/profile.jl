# Main model tests
# ================
using Test, DataFrames, CSV, Distributions, Random
using Beluga
Random.seed!(333)


df1 = CSV.read("test/data/plants1-100.tsv", delim=",")
s1 = open("test/data/plants1.nw", "r") do f ; readline(f); end
shouldbe1 = [-1296.1763022581217, -1408.4459849625102, -1172.356609929616, -1197.7833487345943, -3849.887791929266, -1348.7468163817248, -1486.4818702521393, -2083.8730465634676, -2351.3561862674096, -1364.8837334043392]


@testset "Profile DL model, full loglikelihood" begin
    # test data based on previous Beluga implementation
    λ = [0.56, 0.97, 0.12, 0.21, 4.23, 0.12, 0.77, 2.01, 2.99, 0.44]
    μ = [0.64, 0.36, 0.25, 0.57, 0.42, 0.49, 0.61, 0.69, 3.92, 0.53]
    η = [0.55, 0.56, 0.93, 0.57, 0.59, 0.06, 0.15, 0.41, 0.96, 0.12]
    for i=1:length(λ)
        d, y = DuplicationLossWGDModel(s1, df1, λ[i], μ[i], η[i])
        p = Profile(y)
        l = logpdf!(d, p)
        @test shouldbe1[i] ≈ l
    end
end


@testset "Profile DL model, extending and shrinking" begin
    # test data based on previous Beluga implementation
    λ = [0.56, 0.97, 0.12, 0.21, 4.23, 0.12, 0.77, 2.01, 2.99, 0.44]
    μ = [0.64, 0.36, 0.25, 0.57, 0.42, 0.49, 0.61, 0.69, 3.92, 0.53]
    η = [0.55, 0.56, 0.93, 0.57, 0.59, 0.06, 0.15, 0.41, 0.96, 0.12]
    for i=1:length(λ)
        d, y = DuplicationLossWGDModel(s1, df1, λ[i], μ[i], η[i])
        p = Profile(y)
        l = logpdf!(d, p)
        r = rand(2:length(d), 10)
        wnodes = []
        while length(wnodes) != length(r) || length(r) != 0
            u = rand()
            if length(r) > 0 && u < 0.5
                j = pop!(r)
                w = insertwgd!(d, d[j], rand()*d[j][:t], rand())
                extend!(p, j)
                logpdf!(d[j], p)
                push!(wnodes, w)
            elseif length(wnodes) > 0 && u > 0.5
                wgdnode = pop!(wnodes)
                child = removewgd!(d, wgdnode)
                shrink!(p, wgdnode.i)
                logpdf!(child, p)
            end
        end
        # @show length(d), length(postwalk(d[1]))
        l = logpdf!(d, p)
        @test shouldbe1[i] ≈ l
    end
end
