# Reproduce the walkthrough from Section 6.1 of the MCBench paper
# =================================================================
#
# Run a manageable preview from the repository root with:
#
#     julia --project=. examples/paper_section_6_1.jl
#
# To use the paper's 50 batches of 100,000 samples and its BAT-MH settings:
#
#     julia --project=. examples/paper_section_6_1.jl --paper
#
# The paper configuration is computationally expensive, particularly for MMD.
# Both modes generate the same figures below `examples/paper_section_6_1_output`.
# In addition to the original paper figures, the script demonstrates quantiles,
# analytical reference-value plots, and a combined numerical summary.

using Random

using BAT
using Plots

import MCBench


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

paper_run = true# "--paper" in ARGS
seed = 6101

# Section 6.1 uses 50 independent metric values, each calculated from a batch
# of 100,000 samples. The preview keeps the workflow identical but much smaller.
n_repetitions = paper_run ? 20 : 3
samples_per_repetition = paper_run ? 100_000 : 200
mh_steps = paper_run ? 100_000 : 500
mh_chains = paper_run ? 10 : 2
ess_pilot_runs = paper_run ? 5 : 1

# The metric constructors cap the number of points used by the expensive
# two-sample distances. No overrides are used in paper mode.
swd_metric = paper_run ?
    MCBench.sliced_wasserstein_distance() :
    MCBench.sliced_wasserstein_distance(0.0, samples_per_repetition)
mmd_metric = paper_run ?
    MCBench.maximum_mean_discrepancy() :
    MCBench.maximum_mean_discrepancy(0.0, samples_per_repetition)

output_root = abspath(joinpath(@__DIR__, "paper_section_6_1_output"))
mkpath(output_root)
cd(output_root)

println(paper_run ? "Running the full Section 6.1 configuration." : "Running the quick Section 6.1 preview.")
println("Outputs will be written to ", output_root)


# ---------------------------------------------------------------------------
# Testcase, metrics, and sampler from Section 6.1
# ---------------------------------------------------------------------------

# This built-in testcase is the unimodal Nonlinear 5D Mixture-Laplace-t target
# described in the paper: mode_sep1 = 0 and omega = 0.2.
testcase = MCBench.nonlinear_5d_mixture_laplace_t_easy
parameter_names = ["x1", "x2", "x3", "x4", "x5"]

mean_metric = MCBench.marginal_mean()
variance_metric = MCBench.marginal_variance()
quantile_metric = MCBench.marginal_quantiles() # 50%, 90%, and 99%

# Keep the metrics used by the paper separate so Figure 3 remains a faithful
# reproduction. `metrics` adds the quantiles for the extended MCBench outputs.
paper_metrics = MCBench.TestMetric[
    mean_metric,
    variance_metric,
    swd_metric,
    mmd_metric,
]
metrics = MCBench.TestMetric[
    mean_metric,
    variance_metric,
    quantile_metric,
    swd_metric,
    mmd_metric,
]

sampler = MCBench.BATMH(n_steps=mh_steps, nchains=mh_chains)


# ---------------------------------------------------------------------------
# Figure 2a: marginal x1 distribution for IID and BAT-MH samples
# ---------------------------------------------------------------------------

Random.seed!(seed)
iid_samples = MCBench.sample(
    testcase;
    n_steps=samples_per_repetition,
)
mh_samples = MCBench.sample(testcase, sampler)

# IID samples are stored as vectors. BAT's shaped samples store the same five
# coordinates in the `x` field of a named tuple.
iid_x1 = first.(iid_samples.v)
mh_x1 = first.(getproperty.(mh_samples.v, :x))
x1_bins = range(-18, 18; length=121)

p = stephist(
    iid_x1;
    bins=x1_bins,
    normalize=:pdf,
    color=:blue,
    linewidth=1.5,
    label="IID",
    xlabel="x1",
    ylabel="p(x1)",
    title="Unimodal Nonlinear 5D Mixture-Laplace-t",
    size=(650, 420),
    dpi=300,
)
stephist!(
    p,
    mh_x1;
    bins=x1_bins,
    weights=mh_samples.weight,
    normalize=:pdf,
    color=:red,
    linewidth=1.5,
    label="MH",
)
savefig(p, output_root * "/figure-2a-x1-marginal.pdf")
savefig(p, output_root * "/figure-2a-x1-marginal.png")


# ---------------------------------------------------------------------------
# Generate the repeated metric distributions
# ---------------------------------------------------------------------------

# A sampler build with `unweight=true` (the default) keeps every BAT-MH draw
# intact and creates the IID comparison using a fixed size selected from the
# smallest autocorrelation ESS across its five dimensions in preliminary
# draws. The call therefore writes both statistic distributions and the ESS of
# every benchmark repetition.
Random.seed!(seed + 1)
MCBench.build_teststatistic(
    testcase,
    metrics;
    s=sampler,
    n=n_repetitions,
    n_samples=samples_per_repetition,
    par=false,
    clean=true,
    use_sampler=true,
    ess_pilot_runs=ess_pilot_runs,
    verbose=true,
)

effective_sample_sizes = MCBench.read_effective_sample_sizes(testcase, sampler)
matched_iid_size = MCBench.read_matched_iid_sample_size(testcase, sampler)
println("BAT-MH effective sample sizes: ", effective_sample_sizes)
println("Matched IID sample size: ", matched_iid_size)


# ---------------------------------------------------------------------------
# Figure 3: normalized overview of all metrics
# ---------------------------------------------------------------------------

p = MCBench.plot_metrics(
    testcase,
    paper_metrics,
    sampler;
    names=parameter_names,
    save_plots=false,
)
plot!(
    p;
    title="IID vs. BAT-MH",
    size=(700, 460),
    left_margin=12Plots.mm,
    right_margin=8Plots.mm,
    bottom_margin=8Plots.mm,
)
savefig(p, output_root * "/figure-3-metric-overview.pdf")
savefig(p, output_root * "/figure-3-metric-overview.png")


# ---------------------------------------------------------------------------
# Figure 4a: distribution of the estimated marginal mean of x1
# ---------------------------------------------------------------------------

mean_plots = MCBench.plot_teststatistic(
    testcase,
    mean_metric,
    sampler;
    nbins=20,
    # Separate bins keep both distributions visible in the deliberately small
    # preview. The full run uses common bins, as in the paper comparison.
    same_bins=paper_run,
    show_reference=false,
    show_ks_test=false,
    save_plots=false,
)
p_mean_x1 = mean_plots[1]
plot!(
    p_mean_x1;
    title="Marginal mean of x1",
    size=(520, 360),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p_mean_x1, output_root * "/figure-4a-mean-x1.pdf")
savefig(p_mean_x1, output_root * "/figure-4a-mean-x1.png")


# ---------------------------------------------------------------------------
# Figure 4b: distribution of the sliced Wasserstein distance
# ---------------------------------------------------------------------------

swd_plots = MCBench.plot_teststatistic(
    testcase,
    swd_metric,
    sampler;
    nbins=20,
    same_bins=paper_run,
    show_reference=false,
    show_ks_test=false,
    save_plots=false,
)
p_swd = first(swd_plots)
plot!(
    p_swd;
    title="Sliced Wasserstein distance",
    size=(520, 360),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p_swd, output_root * "/figure-4b-sliced-wasserstein.pdf")
savefig(p_swd, output_root * "/figure-4b-sliced-wasserstein.png")

# Save the two panels together as they appear in Figure 4 of the paper.
p = plot(
    p_mean_x1,
    p_swd;
    layout=(1, 2),
    size=(1100, 420),
    left_margin=6Plots.mm,
    right_margin=4Plots.mm,
    bottom_margin=8Plots.mm,
    dpi=300,
)
savefig(p, output_root * "/figure-4-combined.pdf")
savefig(p, output_root * "/figure-4-combined.png")


# ---------------------------------------------------------------------------
# Additional MCBench output: default marginal quantiles
# ---------------------------------------------------------------------------

# The default quantile metric calculates the 50%, 90%, and 99% quantile for
# every parameter. This overview contains the paper metrics plus all 15
# marginal quantiles and therefore complements, rather than replaces, Figure 3.
p = MCBench.plot_metrics(
    testcase,
    metrics,
    sampler;
    names=parameter_names,
    save_plots=false,
)
plot!(
    p;
    title="Extended IID vs. BAT-MH comparison",
    size=(780, 910),
    left_margin=15Plots.mm,
    right_margin=8Plots.mm,
    bottom_margin=8Plots.mm,
)
savefig(p, output_root * "/additional-metrics-with-quantiles.pdf")
savefig(p, output_root * "/additional-metrics-with-quantiles.png")

# `plot_teststatistic` returns one plot per quantile and parameter. Results are
# ordered by probability and then by parameter. The first 90% result follows
# the five 50% results and therefore belongs to x1.
quantile_plots = MCBench.plot_teststatistic(
    testcase,
    quantile_metric,
    sampler;
    nbins=20,
    same_bins=paper_run,
    show_reference=false,
    show_ks_test=false,
    save_plots=false,
)
p = quantile_plots[testcase.dim + 1]
plot!(
    p;
    title="90% marginal quantile of x1",
    xlabel="90% marginal quantile",
    size=(520, 360),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p, output_root * "/additional-quantile-90-x1.pdf")
savefig(p, output_root * "/additional-quantile-90-x1.png")


# ---------------------------------------------------------------------------
# Additional MCBench output: analytical reference values
# ---------------------------------------------------------------------------

# The testcase stores analytical means and variances. Exact quantile references
# are included wherever available: all medians and the 90% and 99% quantiles of
# normal x1. SWD and MMD are omitted because their ideal score of zero is not a
# target-distribution reference value. Other individual outputs without a
# reference are omitted as well.
p = MCBench.plot_reference_metrics(
    testcase,
    metrics,
    sampler;
    names=parameter_names,
    save_plots=false,
)
plot!(
    p;
    title="BAT-MH vs. reference values",
    legend=:outertopright,
    size=(900, 670),
    left_margin=13Plots.mm,
    right_margin=8Plots.mm,
    bottom_margin=8Plots.mm,
)
savefig(p, output_root * "/additional-reference-metrics.pdf")
savefig(p, output_root * "/additional-reference-metrics.png")


# ---------------------------------------------------------------------------
# Numerical summaries
# ---------------------------------------------------------------------------

# This single table includes every metric, all quantile rows, the empirical IID
# comparison, and every available analytical reference value. A dash in the
# Reference column means that this particular metric output has no analytical
# value, while other outputs of the same metric can still show one.
println()
MCBench.print_metric_summary(
    testcase,
    metrics,
    sampler;
    names=parameter_names,
    include_reference=true,
)

println()
println("Finished. Generated Figures 2a, 3, and 4 plus the extended outputs in ", output_root)
