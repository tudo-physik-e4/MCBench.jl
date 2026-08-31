# MCBench quick start
# ===================
#
# Run this top-to-bottom tutorial from the repository root:
#
#     julia --project=. examples/quickstart.jl
#
# The generated statistics and plots are written to
# `examples/quickstart_output`.

using LinearAlgebra
using Random
using Statistics

using Distributions
using Plots

import MCBench


# Small values keep the tutorial quick. Increase them for a real benchmark.
seed = 2026
n_repetitions = 20
samples_per_repetition = 2000
output_root = abspath(joinpath(@__DIR__, "quickstart_output"))

rng = MersenneTwister(seed)
mkpath(output_root)
cd(output_root)


# ---------------------------------------------------------------------------
# Shared setup: select a testcase and several metrics
# ---------------------------------------------------------------------------

# MCBench provides this three-dimensional standard-normal testcase.
testcase = MCBench.normal_3d_uncorrelated
dimension = testcase.dim
parameter_names = ["x₁", "x₂", "x₃"]

mean_metric = MCBench.marginal_mean()
variance_metric = MCBench.marginal_variance()
skewness_metric = MCBench.marginal_skewness()
sliced_wasserstein_metric = MCBench.sliced_wasserstein_distance(
    0.0,
    samples_per_repetition,
)
mmd_metric = MCBench.maximum_mean_discrepancy(
    0.0,
    samples_per_repetition,
)

metrics = MCBench.TestMetric[
    mean_metric,
    variance_metric,
    skewness_metric,
    sliced_wasserstein_metric,
    mmd_metric,
]

# Known population values are stored directly on provided testcases.
known_mean = MCBench.reference_values(testcase, mean_metric)
known_variance = MCBench.reference_values(testcase, variance_metric)
println("Known target mean: ", known_mean)
println("Known target variance: ", known_variance)


# ---------------------------------------------------------------------------
# Scenario 1: provided target versus external candidate samples
# ---------------------------------------------------------------------------

println("\nScenario 1: provided target versus external candidate samples")

# Pretend this matrix was loaded from another program. Columns are samples and
# rows are parameters. The shifted mean and wider covariance make differences
# visible in the resulting plots.
candidate_distribution = MvNormal(
    [0.35, -0.20, 0.15],
    1.20 .* Matrix{Float64}(I, dimension, dimension),
)
external_candidate_values = rand(
    rng,
    candidate_distribution,
    5 * samples_per_repetition,
)

# Wrap the existing matrix for use by MCBench. This does not generate or alter
# the values. Optional log densities and weights can also be passed here.
external_candidate_dsv = MCBench.make_dsv(external_candidate_values)
external_candidate_sampler = MCBench.DsvSampler(
    [external_candidate_dsv];
    info="External-Candidate",
)

# For samples stored in a CSV file, use this instead:
#
# external_candidate_sampler = MCBench.CsvBasedSampler(
#     "samples.csv";
#     info="External-Candidate",
# )
# MCBench.set_mask(external_candidate_sampler, ["x1", "x2", "x3"])

# First build the IID baseline distribution for every selected metric.
# `use_sampler=false` forwards `n_steps` to each sample call explicitly.
Random.seed!(seed + 1)
baseline_files = MCBench.build_teststatistic(
    testcase,
    metrics;
    n=n_repetitions,
    n_steps=samples_per_repetition,
    n_samples=samples_per_repetition,
    par=false,             # Set true to evaluate metrics concurrently.
    clean=true,            # Replace files from an earlier tutorial run.
    use_sampler=false,
)

# Then build the same distributions for the external candidate sample.
candidate_files = MCBench.build_teststatistic(
    testcase,
    metrics;
    s=external_candidate_sampler,
    n=n_repetitions,
    n_steps=samples_per_repetition,
    n_samples=samples_per_repetition,
    par=false,
    clean=true,
    use_sampler=false,
)

println("  Baseline files: ", baseline_files)
println("  Candidate files: ", candidate_files)

# Persisted results can be loaded as a matrix. Rows are metric outputs and
# columns are repetitions.
candidate_means = MCBench.read_teststatistic(
    testcase,
    mean_metric,
    external_candidate_sampler,
)
println(
    "  Mean over candidate repetitions: ",
    round.(vec(mean(candidate_means; dims=2)); digits=3),
)

# A text summary exposes the actual means, standard deviations, differences,
# and the normalized values shown by the overview plot.
MCBench.print_metric_summary(
    testcase,
    metrics,
    external_candidate_sampler;
    names=parameter_names,
)

# Reference mode keeps metrics with known population values. Its final column
# expresses the raw difference in sampler standard deviations.
MCBench.print_metric_summary(
    testcase,
    metrics,
    external_candidate_sampler;
    names=parameter_names,
    comparison=:reference,
)

scenario_1_plot_dir = joinpath(output_root, testcase.info)
mkpath(scenario_1_plot_dir)

# Individual statistic plots are useful for inspecting distribution shapes.
# As a concise example, save only the first parameter's marginal-mean plot.
p = first(MCBench.plot_teststatistic(
    testcase,
    mean_metric,
    external_candidate_sampler;
    nbins=8,
    show_reference=true,
    save_plots=false,
))
savefig(p, scenario_1_plot_dir * "/mean-x1.pdf")

# MMD has one output for the complete multivariate sample, so it produces one
# plot rather than one plot per parameter.
p = first(MCBench.plot_teststatistic(
    testcase,
    mmd_metric,
    external_candidate_sampler;
    nbins=8,
    save_plots=false,
))
savefig(p, scenario_1_plot_dir * "/mmd.pdf")

# The standard overview contains every selected metric and compares it with the
# empirical IID metric distribution.
p = MCBench.plot_metrics(
    testcase,
    metrics,
    external_candidate_sampler;
    names=parameter_names,
    save_plots=false,
)
savefig(p, scenario_1_plot_dir * "/metrics-overview.pdf")
savefig(p, scenario_1_plot_dir * "/metrics-overview.png")

# This separate overview keeps only metrics with known reference values. Its
# points are differences from those references normalized by their standard
# errors across benchmark repetitions. Every horizontal error bar spans ±1 SEM.
p = MCBench.plot_reference_metrics(
    testcase,
    metrics,
    external_candidate_sampler;
    names=parameter_names,
    save_plots=false,
)
savefig(p, scenario_1_plot_dir * "/reference-metrics-overview.pdf")
savefig(p, scenario_1_plot_dir * "/reference-metrics-overview.png")

println("  Scenario 1 plots saved to: ", scenario_1_plot_dir)


# ---------------------------------------------------------------------------
# Scenario 2: external reference samples versus external candidate samples
# ---------------------------------------------------------------------------

println("\nScenario 2: external reference versus external candidate samples")

# If no sampleable Julia target is available, a trusted external sample can be
# used as the reference distribution. These values are generated only to keep
# the tutorial self-contained; normally they would be loaded from a file.
external_reference_distribution = MvNormal(
    zeros(dimension),
    Matrix{Float64}(I, dimension, dimension),
)
external_reference_values = rand(
    rng,
    external_reference_distribution,
    5 * samples_per_repetition,
)
external_reference_dsv = MCBench.make_dsv(external_reference_values)
external_reference_sampler = MCBench.DsvSampler(
    [external_reference_dsv];
    info="External-Reference-Samples",
)
external_reference_testcase = MCBench.DsvTestcase(
    external_reference_sampler,
    dimension,
    "External-Reference-Normal-3D",
)

# Build empirical baseline distributions by resampling the external reference.
Random.seed!(seed + 2)
external_baseline_files = MCBench.build_teststatistic(
    external_reference_testcase,
    metrics;
    n=n_repetitions,
    n_steps=samples_per_repetition,
    n_samples=samples_per_repetition,
    par=false,
    clean=true,
    use_sampler=false,
)

# Compare the external candidate sample with that empirical reference.
external_candidate_files = MCBench.build_teststatistic(
    external_reference_testcase,
    metrics;
    s=external_candidate_sampler,
    n=n_repetitions,
    n_steps=samples_per_repetition,
    n_samples=samples_per_repetition,
    par=false,
    clean=true,
    use_sampler=false,
)

println("  Baseline files: ", external_baseline_files)
println("  Candidate files: ", external_candidate_files)

# For this scenario the all-metric overview is usually sufficient. There are
# no analytical reference values, so the empirical mean of the external
# reference statistics is used as zero.
scenario_2_plot_dir = joinpath(output_root, external_reference_testcase.info)
mkpath(scenario_2_plot_dir)

p = MCBench.plot_metrics(
    external_reference_testcase,
    metrics,
    external_candidate_sampler;
    names=parameter_names,
    save_plots=false,
)
savefig(p, scenario_2_plot_dir * "/metrics-overview.pdf")
savefig(p, scenario_2_plot_dir * "/metrics-overview.png")

println("  Scenario 2 plots saved to: ", scenario_2_plot_dir)

closeall()

println("\nPlots and statistic files have been written to:")
println(output_root)
