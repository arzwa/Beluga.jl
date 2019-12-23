using CSV, Beluga, DataFrames
using PhyloTree, Luxor, ColorSchemes, Parameters, Printf
import Luxor: RGB
include("src/plot.jl")

begin
    trace = CSV.read("/home/arzwa/rjumpwgd/data/dicots/irmcmc_2223618/trace.csv")
    treefile = "test/data/sim100/plants2.nw"
    nw = open(treefile, "r") do f ; readline(f); end
    model, data = DLWGD(nw, 1., 1., 0.9, Branch)
end

begin
    trace = CSV.read("/home/arzwa/rjumpwgd/data/monocots/irmcmc_2224671/trace.csv")
    treefile = "test/data/monocots/monocots.nw"
    nw = open(treefile, "r") do f ; readline(f); end
    model, data = DLWGD(nw, 1., 1., 0.9, Branch)
end


begin
    w, h = 400, 300
    d = Drawing(w, h, :svg, "/home/arzwa/rjumpwgd/img/dicots-doubletree.svg");
    sethue("black")
    setline(3)
    origin()
    tl1, tl2 = TreeLayout(trace, model)
    xmn, xmx, ymn, ymx = dimensions(tl1)

    @show Luxor.getscale() Luxor.gettranslation()
    Luxor.scale(0.3w/(xmx-xmn),0.45h/(ymx-ymn))
    Luxor.translate(-(xmx-xmn), -(ymx-ymn))

    drawtree(tl1)
    Luxor.translate(2.2*(xmx-xmn), 0.0)
    Luxor.scale(-1,1)
    drawtree(tl2)
    Luxor.translate(1.19*(xmx-xmn), -0.0)
    sethue("black")
    setfont("mono", 12)
    leaflabels(tl1, chain.model.leaves)

    Luxor.translate(-1.3xmx, 0.5ymx)
    Luxor.scale(0.1,4)
    cb = get_colorbar()
    draw_colorbar(cb, exp.(tl1.range)...)

    finish()
    preview()
end
