# List of test cases
The following table contains all test cases currently available in the benchmark suite.
When implementing one of these into the MC sampling framework of your choice, you can use the given testpoints to validate your implementation.  
We provide example implementations of the listed test cases to be used with Julia, Python, R and Stan.  
*This table is not yet complete and will be extended*   

## Reference values

Testcases can store known population values alongside their target definition:

```julia
testcase = Testcases(
    MvNormal(zeros(2), I(2)),
    NamedTupleDist(x=fill(-10..10, 2)),
    2,
    "Normal-2D";
    reference_values=(
        marginal_mean=0.0,
        marginal_variance=[1.0, 1.0],
    ),
)
```

Keys match metric type names. Scalar entries are broadcast across all output
dimensions, while vectors are checked against the metric output dimension. A
configurable metric may use a function entry that receives the metric and
returns the matching values.
`reference_values(testcase, marginal_mean())` returns the normalized vector or
`nothing` when that reference is not defined.

| Name                            | Equation                                                                                                                                                                                                                   | Parameters                                | Testpoints                                                                                        | Julia | Python | R   | Stan |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------ | ----- | ------ | --- | ---- |
| Standard Normal 1D              | $f(x\|\mu, \sigma) =\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$                                                                                                                                      | $\mu = 0, \sigma = 1$                     | $f(x=0) = 0.39894228$, $f(x=1) = 0.24197072$                                                     | ✅     |    ✅     |  ✅    |   ✅    |
| Standard Normal 2D Uncorrelated | $f(x\| \boldsymbol\mu, \boldsymbol\Sigma) = (2\pi)^{-k/2}\det (\boldsymbol\Sigma)^{-1/2} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol\mu)^\mathrm{T} \boldsymbol\Sigma^{-1}(\mathbf{x} - \boldsymbol\mu) \right)$ | $k=2, \mu=\texttt{zeros(2)}, \Sigma= I_2$ | $f(x=[0, 0]) = 0.15915494$, $f(x=[0, 1]) = 0.096532352$, $f(x=[-1, 1]) = 0.0585498315$           | ✅     |   ✅      |  ✅    |   ✅    |
| Standard Normal 3D Uncorrelated | $f(x\| \boldsymbol\mu, \boldsymbol\Sigma) = (2\pi)^{-k/2}\det (\boldsymbol\Sigma)^{-1/2} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol\mu)^\mathrm{T} \boldsymbol\Sigma^{-1}(\mathbf{x} - \boldsymbol\mu) \right)$ | $k=3, \mu=\texttt{zeros(3)}, \Sigma= I_3$ | $f(x=[0, 0, 0]) = 0.063493636$, $f(x=[1, 1, 1]) = 0.014167345$, $f(x=[-1, 0, 1]) = 0.0233580033$ | ✅     |  ✅       |  ✅    |   ✅    |




# List of metrics
The following metrics are available to compare custom generated MC samples to IID samples.

## One-sample metrics
- Marginal mean: `marginal_mean()`
- Marginal variance: `marginal_variance()`
- Marginal quantiles: `marginal_quantiles()` for 50%, 90%, and 99% by default;
  for example, use `marginal_quantiles(percent=95)` for the 95% quantile.
- Global mode: `global_mode()`
- Marginal mode: `marginal_mode()`
- Marginal skewness: `marginal_skewness()`
- Marginal kurtosis: `marginal_kurtosis()`

## Two-sample metric
- Chi-squared: `chi_squared()`
- Sliced Wasserstein Distance: `sliced_wasserstein_distance()`
- Maximum Mean Discrepancy: `maximum_mean_discrepancy()`
