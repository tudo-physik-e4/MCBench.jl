module MCBench

using BAT
using DensityInterface
using Distances
using Distributions
using Folds
using IntervalSets
using JSON
using LinearAlgebra
using Plots
using Random
using Statistics
using StatsBase
using ValueShapes

import DensityInterface: logdensityof

# Core types and low-level sample handling.
include("samplers.jl")
include("testcases.jl")
include("sample_utils.jl")

# Metric implementations and their reference-value interface.
include("mmd.jl")
include("wasserstein.jl")
include("testmetrics.jl")
include("reference_values.jl")

# Benchmark orchestration, persistence, and presentation.
include("twosampleteststatics.jl")
include("teststatistic.jl")
include("metric_summary.jl")
include("plotting_teststat.jl")
include("batmh.jl")

# Built-in benchmark definitions are retained for backward compatibility.
include("../examples/example_distributions.jl")

end # module MCBench
