using Documenter
using Altro 

makedocs(
    sitename = "Altro",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Getting Started" => "quickstart.md",
        "Solver Options" => "options.md",
        "Advanced Options" => "advanced.md",
        "API" => "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RoboticExplorationLab/Altro.jl.git",
)
