# Reproduce the walkthrough from Section 6.2 of the MCBench paper
# =================================================================
#
# Run a manageable preview from the repository root with:
#
#     julia --project=. examples/paper_section_6_2.jl
#
# To use the paper's 50 batches of 100,000 samples and its BAT-MH settings:
#
#     julia --project=. examples/paper_section_6_2.jl --paper
#
# Current MCBench matches the IID batch size to the sampler's estimated ESS.
# To use equal raw batch sizes exactly as described in the original paper:
#
#     julia --project=. examples/paper_section_6_2.jl --paper --raw-batches
#
# The paper configuration is computationally expensive, particularly for MMD.
# Both modes generate the same figures below `examples/paper_section_6_2_output`.
# The experiment follows Figure 2b and Figure 5 from the paper, but intentionally
# uses MCBench's current plots rather than recreating their historical style.
# The updated plots add quantiles, scalar reference values and their p-values,
# sampler-vs-IID KS tests, numerical summaries, and an analytic Laplace
# reference curve for a nonlinear conditional observable.
#
# Output naming guide:
# - `figure-*` files are modern, extended versions of plots in Section 6.2.
# - `additional-*` files demonstrate features added to MCBench afterwards.
# The experiment behind the `figure-*` files is still the one from the paper;
# only the presentation and reported diagnostics have been modernized.

using Random

using BAT
using Distributions
using Plots

import MCBench


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

paper_run = true#"--paper" in ARGS
raw_paper_batches = false# "--raw-batches" in ARGS
seed = 6201

# Section 6.2 uses 50 independent metric values, each calculated from a batch
# of 100,000 samples. The preview keeps the workflow identical but much smaller.
n_repetitions = paper_run ? 50 : 3
samples_per_repetition = paper_run ? 500_000 : 200
mh_steps = paper_run ? 500_000 : 5_000
# Keep all ten chains even in preview mode. With only two chains, BAT's
# convergence check can be unreliable for the two widely separated x1 modes.
mh_chains = 10
ess_pilot_runs = paper_run ? 10 : 1
visual_sample_count = paper_run ? 500_000 : 5_000

# The metric constructors cap the number of points used by the expensive
# two-sample distances. No overrides are used in paper mode.
swd_metric = paper_run ?
    MCBench.sliced_wasserstein_distance() :
    MCBench.sliced_wasserstein_distance(0.0, samples_per_repetition)
mmd_metric = paper_run ?
    MCBench.maximum_mean_discrepancy() :
    MCBench.maximum_mean_discrepancy(0.0, samples_per_repetition)

output_root = abspath(joinpath(@__DIR__, "paper_section_6_2_output"))
mkpath(output_root)
cd(output_root)

println(paper_run ? "Running the full Section 6.2 configuration." : "Running the quick Section 6.2 preview.")
println(raw_paper_batches ?
    "Using equal raw sampler and IID batch sizes, as in the paper." :
    "Matching the IID batch size to the BAT-MH effective sample size.")
println("Outputs will be written to ", output_root)


# ---------------------------------------------------------------------------
# Testcase, metrics, and sampler from Section 6.2
# ---------------------------------------------------------------------------

# This is the hard Nonlinear 5D Mixture-Laplace-t target from Section 6.2. Its
# x1 marginal is an unequal Gaussian mixture with modes at -8 and +8, and its
# faster nonlinear x1-x2 relation uses omega = 0.7. All other parameters use
# the stronger default configuration defined in `example_distributions.jl`.
testcase = MCBench.nonlinear_5d_mixture_laplace_t_hard
parameter_names = ["x1", "x2", "x3", "x4", "x5"]

mean_metric = MCBench.marginal_mean()
variance_metric = MCBench.marginal_variance()
quantile_metric = MCBench.marginal_quantiles() # 50%, 90%, and 99%

# Section 6.2 used means, variances, SWD, and MMD. The updated overview also
# includes the default marginal quantiles to demonstrate the current metric set.
metrics = MCBench.TestMetric[
    mean_metric,
    variance_metric,
    quantile_metric,
    swd_metric,
    mmd_metric,
]

# This benchmark deliberately asks a local random-walk sampler to explore two
# distant modes. Current BAT versions may therefore abort during their own
# between-chain convergence check before MCBench can measure and display the
# failure. `AssumeConvergence` only disables that early abort; proposal tuning,
# burn-in, sampling, ESS estimation, and all MCBench diagnostics remain active.
function hard_target_batmh(; n_steps, nchains)
    algorithm = if isdefined(BAT, :TransformedMCMC)
        BAT.TransformedMCMC(
            proposal=BAT.RandomWalk(),
            nsteps=n_steps,
            nchains=nchains,
            convergence=BAT.AssumeConvergence(),
        )
    else
        BAT.MCMCSampling(
            mcalg=BAT.MetropolisHastings(),
            nsteps=n_steps,
            nchains=nchains,
            convergence=BAT.AssumeConvergence(),
        )
    end
    MCBench.BATMH(algorithm, "BAT-MH")
end

sampler = hard_target_batmh(n_steps=mh_steps, nchains=mh_chains)


# ---------------------------------------------------------------------------
# Updated Figure 2b: marginal x1 distribution for IID and BAT-MH samples
# ---------------------------------------------------------------------------

Random.seed!(seed)
iid_samples = MCBench.sample(
    testcase;
    n_steps=visual_sample_count,
)
visual_sampler = hard_target_batmh(
    n_steps=visual_sample_count,
    nchains=mh_chains,
)
mh_samples = MCBench.sample(testcase, visual_sampler)

# IID samples are stored as vectors. BAT's shaped samples store the same five
# coordinates in the `x` field of a named tuple.
iid_x1 = first.(iid_samples.v)
mh_x1 = first.(getproperty.(mh_samples.v, :x))
x1_bins = range(-22, 22; length=141)

# This marginal overlay remains a visual diagnostic. A naive KS p-value on the
# raw MCMC draws would assume independent observations and would therefore be
# misleading. Figure 5 tests the repeated, ESS-matched metric values instead.
p = stephist(
    iid_x1;
    bins=x1_bins,
    normalize=:pdf,
    color=:blue,
    linewidth=1.5,
    label="IID",
    xlabel="x1",
    ylabel="p(x1)",
    title="Multimodal Nonlinear 5D Mixture-Laplace-t",
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
savefig(p, output_root * "/figure-2b-x1-marginal.pdf")
savefig(p, output_root * "/figure-2b-x1-marginal.png")


# ---------------------------------------------------------------------------
# Generate the repeated metric distributions
# ---------------------------------------------------------------------------

# The paper used 50 sampler batches and 50 IID batches containing 100,000 raw
# samples each. Current MCBench can instead match the IID batch size to the
# sampler's effective sample size, which gives a fairer precision comparison
# for autocorrelated MCMC output. Both workflows are kept explicit here.
if raw_paper_batches
    Random.seed!(seed + 1)
    MCBench.build_teststatistic(
        testcase,
        metrics;
        n=n_repetitions,
        n_steps=samples_per_repetition,
        n_samples=samples_per_repetition,
        par=false,
        clean=true,
        use_sampler=false,
        verbose=true,
    )

    Random.seed!(seed + 2)
    MCBench.build_teststatistic(
        testcase,
        metrics;
        s=sampler,
        n=n_repetitions,
        n_samples=samples_per_repetition,
        unweight=false,
        par=false,
        clean=true,
        use_sampler=true,
        verbose=true,
    )
else
    # With `unweight=true` (the default), this one call writes both the BAT-MH
    # and matched IID statistic distributions, plus the repetition-level ESS.
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
end


# ---------------------------------------------------------------------------
# Figure 5: normalized overview with extended metrics and KS p-values
# ---------------------------------------------------------------------------

p = MCBench.plot_metrics(
    testcase,
    metrics,
    sampler;
    names=parameter_names,
    show_ks_test=true,
    save_plots=false,
)
plot!(
    p;
    xlims=(-5,40),
    title="Extended IID vs. BAT-MH comparison",
    size=(900, 910),
    left_margin=18Plots.mm,
    right_margin=8Plots.mm,
    bottom_margin=8Plots.mm,
)
savefig(p, output_root * "/figure-5-metric-overview.pdf")
savefig(p, output_root * "/figure-5-metric-overview.png")


# ---------------------------------------------------------------------------
# Additional output: distribution of the estimated marginal mean of x1
# ---------------------------------------------------------------------------

mean_plots = MCBench.plot_teststatistic(
    testcase,
    mean_metric,
    sampler;
    nbins=20,
    # Separate bins keep both distributions visible in the deliberately small
    # preview. The full run uses common bins, as in the paper comparison.
    same_bins=paper_run,
    # The dashed line is the known target mean. The title reports both the
    # BAT-MH-vs-IID KS test and the BAT-MH-vs-reference t-test.
    show_reference=true,
    show_reference_test=true,
    show_ks_test=true,
    save_plots=false,
)
p_mean_x1 = mean_plots[1]
plot!(
    p_mean_x1;
    size=(560, 430),
    titlefontsize=9,
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p_mean_x1, output_root * "/additional-mean-x1.pdf")
savefig(p_mean_x1, output_root * "/additional-mean-x1.png")


# ---------------------------------------------------------------------------
# Additional output: distribution of the sliced Wasserstein distance
# ---------------------------------------------------------------------------

swd_plots = MCBench.plot_teststatistic(
    testcase,
    swd_metric,
    sampler;
    nbins=20,
    same_bins=paper_run,
    # SWD has no scalar target reference value. Its title therefore shows the
    # BAT-MH-vs-IID KS test, but correctly does not show a reference t-test.
    show_reference=true,
    show_reference_test=true,
    show_ks_test=true,
    save_plots=false,
)
p_swd = first(swd_plots)
plot!(
    p_swd;
    size=(560, 430),
    titlefontsize=9,
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p_swd, output_root * "/additional-sliced-wasserstein.pdf")
savefig(p_swd, output_root * "/additional-sliced-wasserstein.png")

# A combined version makes the two individual distributions easy to compare.
p = plot(
    p_mean_x1,
    p_swd;
    layout=(1, 2),
    size=(1180, 500),
    left_margin=6Plots.mm,
    right_margin=4Plots.mm,
    bottom_margin=8Plots.mm,
    dpi=300,
)
savefig(p, output_root * "/additional-mean-and-sliced-wasserstein.pdf")
savefig(p, output_root * "/additional-mean-and-sliced-wasserstein.png")


# ---------------------------------------------------------------------------
# Additional MCBench output: one individual quantile distribution
# ---------------------------------------------------------------------------

# `plot_teststatistic` returns one plot per quantile and parameter. Results are
# ordered by probability and then by parameter. In the hard target, exact
# marginal quantiles are only available for the medians of x4 and x5. The
# fourth result is therefore the 50% quantile of x4 and has reference value 0:
#
# - the two-sample KS test compares the repeated BAT-MH and IID metric values;
# - the reference t-test compares the repeated BAT-MH estimates with the known
#   scalar median, using their standard error of the mean.
quantile_plots = MCBench.plot_teststatistic(
    testcase,
    quantile_metric,
    sampler;
    nbins=20,
    same_bins=paper_run,
    show_reference=true,
    show_reference_test=true,
    show_ks_test=true,
    save_plots=false,
)
p = quantile_plots[4]
plot!(
    p;
    xlabel="50% marginal quantile",
    size=(640, 430),
    left_margin=6Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p, output_root * "/additional-quantile-50-x4.pdf")
savefig(p, output_root * "/additional-quantile-50-x4.png")


# ---------------------------------------------------------------------------
# Additional MCBench output: analytical reference values
# ---------------------------------------------------------------------------

# The hard testcase stores analytical means and variances, plus exact medians
# for the symmetric x4 and x5 marginals. Its mixture and nonlinear marginals do
# not have stored analytical quantiles. SWD and MMD are omitted because their
# ideal score of zero is not a target-distribution reference value.
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
# Additional MCBench output: an analytic reference distribution
# ---------------------------------------------------------------------------

# The conditional model for x2 supplies an exact reference distribution that
# belongs to the nonlinear Section 6.2 target itself. In particular,
#
#   x2 | x1 ~ Laplace(A sin(omega x1), b2)
#
# implies that the standardized conditional residual
#
#   T2(x) = (x2 - A sin(omega x1)) / b2
#
# follows Laplace(0, 1). This checks the nonlinear x1-x2 relation rather than
# only one marginal distribution.
target_params = MCBench.nonlinear_5d_mixture_laplace_t_hard_params

function x2_standardized_residual(sample_value)
    x1 = sample_value[1]
    x2 = sample_value[2]
    conditional_mean = target_params.A * sin(target_params.ω * x1)
    (x2 - conditional_mean) / target_params.b2
end

x2_residual_reference = MCBench.AnalyticReferenceDistribution(
    x2_standardized_residual,
    Laplace(0.0, 1.0);
    info="Standardized x2 conditional residual",
)

# Analytic reference distributions are stored separately from scalar reference
# values. Rebuilding the testcase here preserves all scalar references and adds
# the custom observation-level transform used by this demonstration.
testcase_with_reference_curve = MCBench.Testcases(
    testcase.f,
    testcase.bounds,
    testcase.dim,
    testcase.info;
    reference_values=testcase.reference_values,
    reference_distributions=merge(
        testcase.reference_distributions,
        (x2_residual=x2_residual_reference,),
    ),
)

# Reuse the BAT-MH samples plotted in Figure 2b. The test transforms each
# five-dimensional observation and compares the residuals directly with the
# analytic Laplace curve; no IID reference sample is generated. The empirical
# curve and KS statistic use all transformed observations, whereas the KS
# p-value is calibrated with the conservative autocorrelation-based ESS.
analytic_effective_sample_size = MCBench.get_effective_sample_size(
    mh_samples,
    sampler,
)
analytic_result = MCBench.reference_distribution_test(
    testcase_with_reference_curve,
    mh_samples,
    :x2_residual,
    effective_sample_size=analytic_effective_sample_size,
)

println()
println("Nonlinear x2-residual analytic comparison:")
println("  reference distribution = ", analytic_result.reference.distribution)
println("  transformed observations = ", length(analytic_result.observable_values))
println("  effective sample size = ", analytic_result.effective_sample_size)
println("  one-sample KS statistic = ", analytic_result.statistic)
println("  ESS-adjusted one-sample KS p-value = ", analytic_result.pvalue)

# The upper panel compares the empirical transformed distribution with the
# analytic Laplace density. The lower panel compares their CDFs, which are
# the quantities used by the one-sample KS statistic.
#
# Replacing the raw sample count with an autocorrelation ESS is an approximate
# MCMC correction, rather than an exact dependent-sample KS calibration. The
# plot therefore reports both the raw observation count and the ESS used.
p = MCBench.plot_reference_distribution(
    analytic_result;
    # The density carries most of the visual information. Keep the CDF as a
    # compact lower panel because it directly shows the discrepancy used by
    # the KS statistic. Both legend positions can be changed independently.
    # Add `title="..."` here to replace the automatically generated title.
    panel_heights=(4, 1),
    nbins= 200,
    distribution_legend=:topright,
    cdf_legend=:bottomright,
    save_plots=false,
)
plot!(
    p;
    size=(780, 650),
    left_margin=7Plots.mm,
    right_margin=7Plots.mm,
    bottom_margin=7Plots.mm,
)
savefig(p, output_root * "/additional-x2-residual-laplace-reference.pdf")
savefig(p, output_root * "/additional-x2-residual-laplace-reference.png")

# See `examples/analytic_reference_distribution.jl` for the next step: defining
# other custom observables and their analytic reference distributions.


# ---------------------------------------------------------------------------
# Numerical summaries
# ---------------------------------------------------------------------------

# This table includes every metric, all quantile rows, the empirical IID
# comparison, and every available analytical scalar reference value. It also
# reports the sampler-vs-IID KS p-value and, where a scalar reference exists,
# its one-sample reference t-test p-value. A dash means that the particular
# metric output has no known scalar reference.
println()
MCBench.print_metric_summary(
    testcase,
    metrics,
    sampler;
    names=parameter_names,
    include_reference=true,
)

# The reference-only table omits metrics without known scalar references. Its
# normalized difference is (sampler mean - reference) / sampler SEM, which is
# also the t statistic used to calculate the displayed reference p-value.
println()
MCBench.print_metric_summary(
    testcase,
    metrics,
    sampler;
    names=parameter_names,
    comparison=:reference,
)

println()
println(
    "Finished. Generated updated Figure 2b and Figure 5 plus additional outputs in ",
    output_root,
)
