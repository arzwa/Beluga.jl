env_dir = "."
using Pkg; Pkg.activate(env_dir)
using CSV, Beluga, DataFrames, RecipesBase, Distributions
using PhyloTree, Luxor, ColorSchemes, Parameters, Printf
import Luxor: RGB
include("../src/plot.jl")

base = "/home/arzwa/research/rjumpwgd/"

begin
    trace = CSV.read(joinpath(base, "data/dicots/irmcmc_2223618/trace.csv"))
    treefile = "test/data/sim100/plants2.nw"
    nw = open(treefile, "r") do f ; readline(f); end
    model, data = DLWGD(nw, 1., 1., 0.9, Branch)
end

begin
    trace = CSV.read(joinpath(base, "data/monocots/irmcmc_2224671/trace.csv"))
    treefile = "test/data/monocots/monocot.nw"
    nw = open(treefile, "r") do f ; readline(f); end
    model, data = DLWGD(nw, 1., 1., 0.9, Branch)
end

begin
    trace = CSV.read(joinpath(base, "data/hexapods/irmcmc_2230169/trace.csv"))
    treefile = "test/data/hexapods/hexapods3.nw"
    nw = open(treefile, "r") do f ; readline(f); end
    model, data = DLWGD(nw, 1., 1., 0.9, Branch)
end


begin
    w, h = 500, 30*(length(model.leaves))
    d = Drawing(w, h, :svg, "$base/img/dicots-doubletree.svg");
    sethue("black")
    setline(3)
    origin()
    tl1, tl2 = TreeLayout(trace, model)
    xmn, xmx, ymn, ymx = dimensions(tl1)

    @show Luxor.getscale() Luxor.gettranslation()
    Luxor.scale(0.3w/(xmx-xmn),0.45h/(ymx-ymn))
    Luxor.translate(-(xmx-xmn), -(ymx-ymn))

    drawtree!(tl1)
    sethue("black")
    setfont("Noto mono", 9)
    internalnodes!(tl1, model)
    Luxor.translate(2.2*(xmx-xmn), 0.0)
    Luxor.scale(-1,1)
    drawtree!(tl2)
    Luxor.translate(1.17*(xmx-xmn), -0.0)
    sethue("black")
    setfont("Noto mono", 11)
    leaflabels!(tl1, model.leaves)

    Luxor.translate(-1.3xmx, 0.5ymx)
    Luxor.scale(0.1,4)
    cb = getcolorbar()
    drawcolorbar!(cb, exp10.(tl1.range)...)

    finish()
    preview()
end
