# Using MCBench
This is a simple example of how to use the MCBench package. 

For a complete executable version of this workflow, run the heavily commented
[`examples/quickstart.jl`](https://github.com/tudo-physik-e4/MCBench.jl/blob/main/examples/quickstart.jl)
script from the repository root:

```bash
julia --project=. examples/quickstart.jl
```

It additionally demonstrates external candidate samples, an external sample
used as the reference distribution, reference-aware plots, and explicit plot
saving.

## Specify the test case
```
f = MvNormal(zeros(3), I(3))
bounds = NamedTupleDist(x = [-10..10 for i in 1:3])
Standard_Normal_3D_Uncorrelated = Testcases(
    f,
    bounds,
    3,
    "Normal-3D-Uncorrelated";
    reference_values=(
        marginal_mean=zeros(3),
        marginal_variance=ones(3),
    ),
)
```
 
## Selecting the metrics to be applied
```
metrics = [marginal_mean(), marginal_variance(), sliced_wasserstein_distance(), maximum_mean_discrepancy()]
```
 
## Load the external MC samples to be tested
```
sampler = FileBasedSampler("samples_from_my_algorithm.csv")
```

## Generating the test statistics
Evaluate the metrics both

- for the IID samples (IID samples are generated automatically in the background):
```
teststatistics_IID = build_teststatistic(Standard_Normal_3D_Uncorrelated, metrics,
n=100, n_steps=10^5, n_samples=10^5)
```
- and for the MC samples to be tested:
```
teststatistics_my_samples = build_teststatistic(Standard_Normal_3D_Uncorrelated, metrics,
n=100, n_steps=10^5, n_samples=10^5, s=sampler)
```

### Sampling and parallelism options

`par=true` evaluates the different metrics concurrently on each sample batch.
It does not parallelize sampling or the repetitions themselves. This is most
useful when several expensive metrics are selected; it has no effect for a
single metric.

`use_sampler` controls where the configuration for an individual draw comes
from. It does not enable or disable `s`:

- `use_sampler=true` calls `sample(testcase, s)`, using settings stored in the
  sampler or its method defaults.
- `use_sampler=false` calls `sample(testcase, s; n_steps=n_steps)`, explicitly
  forwarding the `n_steps` supplied to `build_teststatistic`.

When `n_samples` is positive, MCBench collects sampler draws until it can form
a batch of exactly that size. Remaining samples are kept for the next
repetition. Set `n_samples <= 0` to evaluate one complete draw per repetition.

There is one intentional exception. For a non-IID `SamplingAlgorithm`, such
as `BATMH`, `unweight=true` enables ESS matching: every complete sampler draw
is retained and MCBench calculates the smallest autocorrelation ESS across its
dimensions. The same call writes the matched IID statistic distribution. Its
fixed batch size is the median of the minimum ESS values from
`ess_pilot_runs` preliminary draws. Inspect the actual repetition ESS values
with `read_effective_sample_sizes(testcase, sampler)` and the fixed size with
`read_matched_iid_sample_size(testcase, sampler)`. Set `unweight=false` to use
ordinary raw-sample batching instead.

## Generating text summaries

Use the text summary to inspect the numbers behind the overview plots:

```julia
print_metric_summary(Standard_Normal_3D_Uncorrelated, metrics, sampler)

print_metric_summary(
    Standard_Normal_3D_Uncorrelated,
    metrics,
    sampler;
    comparison=:reference,
)
```

The reported uncertainty is the empirical standard deviation across benchmark
repetitions. Call `metric_summary(...)` when the formatted table should be
returned as a string without being printed, or `metric_summary_rows(...)` for
the numeric values as named tuples. For IID comparisons, every row also reports
the two-sample KS statistic and its unadjusted p-value. The null hypothesis is
that the repeated sampler and IID metric values follow the same continuous
distribution; a large p-value is not proof that the distributions agree.

## Generating comparison plots
- Overview plot of all selected metrics
```
plot_metrics(Standard_Normal_3D_Uncorrelated, metrics, sampler)
```

- Overview of metrics with known reference values
```
plot_reference_metrics(Standard_Normal_3D_Uncorrelated, metrics, sampler)
```
<img src="../images/Normal-3D-Uncorrelated-metrics.svg" width="480"/>

- Individual metrics
```
plot_teststatistic(Standard_Normal_3D_Uncorrelated, marginal_mean(), sampler; nbins=20)
```
<img src="../images/Normal-3D-Uncorrelated-SlicedWasserstein.svg" width="480"/>

The individual plot automatically includes the known mean as a dashed line.
Use `show_reference=false` when the line is not wanted. `plot_metrics` uses the
historical `(mean(metric) - mean(IID metric)) / std(IID metric)` normalization.
Sampler-versus-IID histogram titles show the KS statistic and p-value computed
from the unbinned values. When a reference is stored, they separately show the
two-sided one-sample t-test against it; use `show_ks_test=false` or
`show_reference_test=false` to hide the corresponding annotation.
`plot_reference_metrics` normalizes every difference from a known value by its
standard error across benchmark repetitions. Each black point is
`(mean(metric) - reference) / SEM(metric)`, and its horizontal error bar spans
±1 SEM on that normalized scale. The row label reports the associated
reference-test p-value.
