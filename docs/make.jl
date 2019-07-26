using Documenter, SIMDIntrinsics

# Generate examples
import Literate

const EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
const GENERATED_DIR = joinpath(@__DIR__, "src", "examples", "generated")
rm(GENERATED_DIR; force=true, recursive=true)

for example in readdir(EXAMPLE_DIR)
    endswith(example, ".jl") || continue
    input = abspath(joinpath(EXAMPLE_DIR, example))
    Literate.markdown(input, GENERATED_DIR)
end

const examples = joinpath.("examples", "generated", filter!(x -> endswith(x, ".md"), readdir(GENERATED_DIR)))

# Build documentation.
makedocs(
    format = Documenter.HTML(prettyurls = haskey(ENV, "HAS_JOSH_K_SEAL_OF_APPROVAL")), # disable for local builds
    sitename = "SIMDIntrinsics.jl",
    doctest = true,
    pages = Any[
        "Home" => "index.md",
        "Examples" => [
            "Examples" => examples,
        ]
    ]
)


# Deploy built documentation from Travis.
deploydocs(
    repo = "github.com/KristofferC/SIMDIntrinsics.jl.git",
)