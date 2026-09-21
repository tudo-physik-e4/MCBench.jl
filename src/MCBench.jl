module MCBench

using Distributions 
using Statistics 
using ValueShapes
using LinearAlgebra
using Folds
using IntervalSets
using BAT
using StatsBase
using Plots
using JSON  #for pasting teststatistic quickly 
using StructArrays
using HypothesisTests
using DensityInterface
import DensityInterface: logdensityof
using Distances 
using DataFrames
using CSV

include("samplers.jl")
include("testcases.jl")
include("mmd.jl")
include("testmetrics.jl")
include("wasserstein.jl")
include("twosampleteststatics.jl")
include("nammd.jl")
include("teststatistic.jl")
include("sample_utils.jl")
include("plotting_teststat.jl")
include("batmh.jl")
include("../examples/example_distributions.jl")
include("../examples/example_posteriordb.jl")

end # module
