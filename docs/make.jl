using Documenter, Beluga

makedocs(
    sitename = "Beluga.jl",
    modules = [Beluga],
    authors = "Arthur Zwaenepoel",
    pages = [
        "Introduction" => "index.md",
        "Reversible-jump MCMC for WGD inference" => "rjmcmc.md",
        "Maximum-likelihood estimation" => "mle.md",
        "API" => "api.md"],
    clean = true)

deploydocs(
    repo = "github.com/arzwa/Beluga.jl.git",
    target = "build")
