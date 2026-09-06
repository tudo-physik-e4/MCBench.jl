# MCBench — Monte Carlo Sampling Benchmark Suite

[![Documentation](https://img.shields.io/badge/Documentation-dev-blue.svg)](https://tudo-physik-e4.github.io/MCBench.jl/dev)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![CI](https://github.com/tudo-physik-e4/MCBench/actions/workflows/ci.yml/badge.svg)](https://github.com/tudo-physik-e4/MCBench/actions/workflows/ci.yml)

MCBench is a Julia package for quantitative comparisons of Monte Carlo
samples. It evaluates statistics such as marginal moments, quantiles, sliced
Wasserstein distance, and maximum mean discrepancy over repeated sample
batches. The resulting distributions, plots, and numerical summaries help
identify bias, underestimated uncertainty, missed modes, and other sampling
problems.

MCBench supports three distinct reference modes:

1. comparison with analytically known scalar values, such as a target mean;
2. comparison with empirical IID samples or another trusted reference sample;
3. comparison of a sample-derived observable with an analytically known
   probability distribution.

Sampling may be performed through BAT.jl or outside Julia. Existing samples can
be supplied as matrices, BAT density sample vectors, or CSV files.

The accompanying paper is available on
[arXiv](https://arxiv.org/abs/2501.03138).

## Installation

MCBench supports Julia 1.9 and newer. Julia can be downloaded from the
[official Julia website](https://julialang.org/downloads/).

### Install MCBench in a Julia environment

Start Julia in the project where you want to use MCBench:

```bash
julia --project=.
```

Then install the package from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/tudo-physik-e4/MCBench.jl.git")
```

The package manager installs MCBench and its dependencies into the active Julia
environment. Verify the installation with:

```julia
import MCBench
MCBench.normal_3d_uncorrelated
```

If you do not want to use a project-specific environment, start Julia without
`--project=.`; the package will then be added to your currently active Julia
environment.

### Clone the repository

Clone the repository when you want to run the included examples, execute the
test suite, or modify MCBench itself:

```bash
git clone https://github.com/tudo-physik-e4/MCBench.jl.git
cd MCBench.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Run the tests from the repository root with:

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

## Quick start

The following workflow compares external samples with IID samples from a
built-in three-dimensional standard-normal target.

### 1. Select a testcase and metrics

```julia
import MCBench

testcase = MCBench.normal_3d_uncorrelated

metrics = MCBench.TestMetric[
    MCBench.marginal_mean(),
    MCBench.marginal_variance(),
    MCBench.marginal_quantiles(),
    MCBench.sliced_wasserstein_distance(),
    MCBench.maximum_mean_discrepancy(),
]
```

`marginal_quantiles()` evaluates the 50%, 90%, and 99% quantiles by default.
For example, `marginal_quantiles(percent=95)` selects only the 95% quantile.

### 2. Load external samples

A `CsvBasedSampler` expects a header row followed by one sample per row. Use
`set_mask` to select parameter columns and put them in testcase order:

```julia
sampler = MCBench.CsvBasedSampler(
    "samples_from_my_algorithm.csv";
    info="My-Sampler",
)
MCBench.set_mask(sampler, ["x1", "x2", "x3"])
```

For samples already held in Julia, columns of the matrix are observations:

```julia
candidate_samples = MCBench.make_dsv(sample_matrix)
sampler = MCBench.DsvSampler([candidate_samples]; info="My-Sampler")
```

### 3. Build repeated metric distributions

First generate the empirical IID baseline:

```julia
MCBench.build_teststatistic(
    testcase,
    metrics;
    n=20,
    n_steps=2_000,
    n_samples=2_000,
    par=false,
    clean=true,
    use_sampler=false,
)
```

Then evaluate the same metrics on the external samples:

```julia
MCBench.build_teststatistic(
    testcase,
    metrics;
    s=sampler,
    n=20,
    n_steps=2_000,
    n_samples=2_000,
    par=false,
    clean=true,
    use_sampler=false,
)
```

Here, `n` is the number of metric repetitions and `n_samples` is the batch
size used for each repetition. With `use_sampler=false`, `n_steps` is passed
explicitly to each sampling call. A sequential CSV source must therefore
contain enough rows for all requested repetitions. Set `par=true` to evaluate
different metrics concurrently within each repetition.

The generated line-delimited JSON files are stored below
`teststatistics/` and `teststatistics_sampler/` in the current working
directory. Set `clean=true` to replace results from an earlier run;
`clean=false` appends new repetitions.

### 4. Inspect the results

Print the numerical comparison:

```julia
MCBench.print_metric_summary(
    testcase,
    metrics,
    sampler;
    names=["x₁", "x₂", "x₃"],
    include_reference=true,
)
```

Create the overall IID comparison and the separate scalar-reference overview:

```julia
MCBench.plot_metrics(testcase, metrics, sampler)
MCBench.plot_reference_metrics(testcase, metrics, sampler)
```

Individual metric distributions can also be inspected:

```julia
MCBench.plot_teststatistic(
    testcase,
    MCBench.marginal_mean(),
    sampler;
    nbins=20,
)
```

Plotting functions save files by default. Pass `save_plots=false` to receive a
Plots.jl object that can be customized before saving.

## Comparison modes

### Known scalar reference values

A testcase may store population values for selected metrics, such as exact
means, variances, or marginal quantiles. They can be queried directly:

```julia
metric = MCBench.marginal_mean()
MCBench.reference_values(testcase, metric)
```

`plot_reference_metrics` compares the mean of each repeated sampler metric
with its known value. The difference is normalized by the sampler metric's
standard error of the mean:

```text
(mean(metric) - reference) / SEM(metric)
```

The horizontal error bar spans ±1 SEM, and the row label reports the
two-sided reference-test p-value. Outputs without a stored reference are
omitted individually.

### Empirical IID or reference samples

`plot_metrics` compares repeated sampler metrics with repeated IID metrics
using the historical MCBench normalization:

```text
(mean(sampler metric) - mean(IID metric)) / std(IID metric)
```

The green, yellow, and red background regions mark one, two, and three IID
standard deviations. Row labels report the two-sample Kolmogorov–Smirnov p-value
for the unbinned repeated metric values.

When a target cannot generate IID samples, wrap a trusted external sample in a
`DsvSampler` and construct a `DsvTestcase`. The
[quickstart script](examples/quickstart.jl) contains a complete example.

### Analytic reference distributions

An `AnalyticReferenceDistribution` combines:

- an observable that maps one multivariate observation to a scalar; and
- the exact distribution of that scalar under the target.

No IID reference sample is generated. For every built-in Gaussian testcase,
the squared Mahalanobis distance is available and follows a chi-squared law:

```julia
samples = MCBench.sample(MCBench.normal_3d_uncorrelated, 10_000)

result = MCBench.reference_distribution_test(
    MCBench.normal_3d_uncorrelated,
    samples,
    :squared_mahalanobis,
)

println(result.statistic)
println(result.pvalue)
MCBench.plot_reference_distribution(result)
```

The upper plot compares the empirical density with `Chisq(3)); the smaller
lower panel compares the empirical and analytic CDFs used by the one-sample KS
test.

For correlated MCMC samples, pass an effective sample size when testing an
existing sample:

```julia
effective_size = MCBench.get_effective_sample_size(mcmc_samples, sampler)

result = MCBench.reference_distribution_test(
    testcase,
    mcmc_samples,
    :observable;
    effective_sample_size=effective_size,
)
```

This keeps the empirical KS statistic unchanged but calibrates its p-value with
the effective rather than raw sample count. It is an approximate correction
for dependence, not an exact dependent-sample KS test. The paper examples show
how to define a custom observable and analytic reference law.

## MCMC and effective sample size

For a non-IID `SamplingAlgorithm`, such as `BATMH`, the default
`unweight=true` enables ESS-matched comparison. Pilot draws estimate the
smallest autocorrelation-based ESS across dimensions. Their median determines
one fixed IID batch size, after which MCBench builds both the sampler and
matched IID statistic distributions in the same call:

```julia
sampler = MCBench.BATMH(n_steps=100_000, nchains=10)

MCBench.build_teststatistic(
    testcase,
    metrics;
    s=sampler,
    n=50,
    clean=true,
    unweight=true,
    ess_pilot_runs=5,
)
```

Inspect the recorded values with:

```julia
MCBench.read_effective_sample_sizes(testcase, sampler)
MCBench.read_matched_iid_sample_size(testcase, sampler)
```

The sampler metric distribution still reflects the run-to-run ESS variation,
while the IID reference uses the single pilot-derived batch size. This is why
sampler error bars and IID bands need not have identical widths.

## Built-in testcases

The complete catalog and its parameter definitions are in
[`src/builtin_testcases.jl`](src/builtin_testcases.jl).

| Family | Available configurations | Main challenge |
| --- | --- | --- |
| Independent normal | 1D, 2D, 3D, 10D, 100D | Baseline correctness and dimensional scaling |
| Equicorrelated normal | Weak and strong correlation in 2D, 10D, 100D | Linear dependence |
| Gaussian mixtures | Univariate and multivariate; equal or unequal mode weights | Separated modes |
| Cauchy | 1D | Heavy tails and undefined moments |
| Nonlinear Mixture–Laplace–t | Easy and hard 5D configurations | Nonlinearity, mixtures, heteroskedasticity, and heavy tails |
| Eight schools | Original and log-scale parameterizations | Hierarchical posterior geometry |

For example:

```julia
easy = MCBench.nonlinear_5d_mixture_laplace_t_easy
hard = MCBench.nonlinear_5d_mixture_laplace_t_hard
eight_schools = MCBench.eight_schools_testcase
```

## Metrics

One-sample metrics operate on one sample batch and usually produce one result
per dimension:

- `marginal_mean()`
- `marginal_variance()`
- `marginal_quantiles()`
- `global_mode()`
- `marginal_mode()`
- `marginal_skewness()`
- `marginal_kurtosis()`

Two-sample metrics compare a candidate batch with an IID or external reference
batch:

- `wasserstein_1d()`
- `sliced_wasserstein_distance()`
- `maximum_mean_discrepancy()`
- `chi_squared()` (legacy standardized discrepancy)

Use Julia's help mode for detailed signatures and keyword descriptions, for
example `?MCBench.build_teststatistic` or `?MCBench.plot_metrics`.

## Examples

The repository intentionally keeps the example set small:

- [`examples/quickstart.jl`](examples/quickstart.jl) is the main tutorial. It
  demonstrates built-in and external samples, persisted statistics, summaries,
  scalar reference values, and plots.
- [`examples/paper_section_6_1.jl`](examples/paper_section_6_1.jl) runs the
  easy nonlinear benchmark.
- [`examples/paper_section_6_2.jl`](examples/paper_section_6_2.jl) repeats the
  analysis for the difficult multimodal configuration.

From a cloned repository, run the quickstart with:

```bash
julia --project=. examples/quickstart.jl
```

The paper scripts run a smaller preview by default. Add `--paper` for their
full settings and `--raw-batches` to reproduce the original raw-batch
comparison without ESS matching:

```bash
julia --project=. examples/paper_section_6_1.jl
julia --project=. examples/paper_section_6_1.jl --paper
julia --project=. examples/paper_section_6_2.jl --paper --raw-batches
```

Outputs are written below the corresponding directory in `examples/`.

<img src="docs/images/MCBench-Workflow.svg" width="800" alt="MCBench workflow"/>

## Further documentation

The development documentation is available at
[tudo-physik-e4.github.io/MCBench.jl/dev](https://tudo-physik-e4.github.io/MCBench.jl/dev).
All public functions used by the examples also provide Julia docstrings.

## License

MCBench is distributed under the [MIT License](LICENSE.md).

