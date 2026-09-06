module MCBench

using BAT
using DensityInterface
using Distances
using Distributions
using Folds
using HypothesisTests
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
include("analytic_reference_distributions.jl")

# Metric implementations and their reference-value interface.
include("mmd.jl")
include("wasserstein.jl")
include("testmetrics.jl")
include("reference_values.jl")

# Benchmark orchestration, persistence, and presentation.
include("twosampleteststatics.jl")
include("teststatistic.jl")
include("statistical_tests.jl")
include("metric_summary.jl")
include("plotting_teststat.jl")
include("batmh.jl")

# Targets distributed with MCBench.
include("builtin_testcases.jl")

end # module MCBench
