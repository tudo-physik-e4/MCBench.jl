"""
    ks_test(first, second)

Perform a two-sided, approximate two-sample Kolmogorov-Smirnov test on two
collections of persisted test-statistic values. The returned named tuple
contains the KS statistic `statistic` and its unadjusted `pvalue`.

The null hypothesis is that both collections were drawn from the same
continuous distribution. A p-value is not the probability that the samples
agree; interpret it together with the KS statistic and the number of benchmark
repetitions.
"""
function ks_test(
    first::AbstractVector{<:Real},
    second::AbstractVector{<:Real},
)
    isempty(first) && throw(ArgumentError("first KS-test input cannot be empty"))
    isempty(second) && throw(ArgumentError("second KS-test input cannot be empty"))
    all(isfinite, first) || throw(ArgumentError("first KS-test input must be finite"))
    all(isfinite, second) || throw(ArgumentError("second KS-test input must be finite"))

    test = ApproximateTwoSampleKSTest(Float64.(first), Float64.(second))
    (statistic=Float64(test.δ), pvalue=Float64(pvalue(test; tail=:both)))
end

export ks_test
