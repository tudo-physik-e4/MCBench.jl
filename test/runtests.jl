ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Test
using Random
using LinearAlgebra
using Statistics
using StatsBase
using Distributions
using DensityInterface
using IntervalSets
using ValueShapes
using BAT

import MCBench

include("fixtures.jl")

@info "Running tests with $(Base.Threads.nthreads()) Julia threads active."

@testset verbose = true "MCBench" begin
    include("testcases.jl")
    include("sample_utils.jl")
    include("samplers.jl")
    include("metrics.jl")
    include("distances.jl")
    include("teststatistics.jl")
    include("plotting.jl")
    include("examples.jl")
end
