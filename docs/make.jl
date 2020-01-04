using Documenter, Beluga

makedocs(
    sitename = "Beluga.jl",
    modules  = [Beluga],
    authors  = "Arthur Zwaenepoel",
    pages    = ["Beluga" => "index.md", "API" => "api.md"],
    clean    = true
)

deploydocs(
    repo = "github.com/arzwa/Beluga.jl.git",
    target = "build",
)
