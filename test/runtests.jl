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
    include("example_distributions.jl")
    include("sample_utils.jl")
    include("analytic_reference_distributions.jl")
    include("samplers.jl")
    include("metrics.jl")
    include("distances.jl")
    include("teststatistics.jl")
    include("statistical_tests.jl")
    include("metric_summary.jl")
    include("plotting.jl")
    include("examples.jl")
    include("end_to_end.jl")
    include("documentation.jl")
end
