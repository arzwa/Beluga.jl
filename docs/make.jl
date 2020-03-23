using Documenter, Beluga, Literate

fnames = String[]
ignore = String[]
outdir = joinpath(@__DIR__, "src")
srcdir = joinpath(@__DIR__, "lit")
mkpath(outdir)
for f in readdir(srcdir)
    if endswith(f, ".jl") && !(startswith(f, "_"))
        target = string(split(f,".")[1])
        outpath = joinpath(outdir, target*".md")
        if f ∈ ignore
            try rm(outpath) ; catch ; end
            continue
        end
        push!(fnames, relpath(outpath, joinpath(@__DIR__, "src")))
        @info "Literating $f"
        Literate.markdown(joinpath(srcdir, f), outdir, documenter=true)
        x = read(`tail -n +4 $outpath`)
        write(outpath, x)
    end
end

makedocs(
    sitename = "Beluga.jl",
    modules = [Beluga],
    authors = "Arthur Zwaenepoel",
    pages = [
        "Introduction" => "index.md",
        "Examples"=>fnames,
        "API" => "api.md"],
    clean = true)

deploydocs(
    repo = "github.com/arzwa/Beluga.jl.git",
    target = "build")
