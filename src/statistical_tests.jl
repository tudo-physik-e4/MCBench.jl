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

"""
    reference_value_test(values, reference)

Perform a two-sided one-sample t-test of repeated metric values against a
stored testcase reference. The test statistic is

`t = (mean(values) - reference) / SEM(values)`,

where `SEM = std(values) / sqrt(length(values))`. The returned named tuple
contains `statistic`, `pvalue`, `standard_error`, and `degrees_of_freedom`.

The test treats benchmark repetitions as independent observations and assumes
that their mean is approximately normally distributed. A p-value is not the
probability that the metric agrees with the reference; it measures how
surprising the observed mean difference would be under exact agreement.
"""
function reference_value_test(
    values::AbstractVector{<:Real},
    reference::Real,
)
    length(values) > 1 || throw(ArgumentError(
        "at least two repeated metric values are required for a reference-value test",
    ))
    all(isfinite, values) || throw(ArgumentError(
        "reference-test values must be finite",
    ))
    isfinite(reference) || throw(ArgumentError("reference value must be finite"))

    float_values = Float64.(values)
    float_reference = Float64(reference)
    sample_mean = mean(float_values)
    standard_error = std(float_values) / sqrt(length(float_values))
    difference = sample_mean - float_reference

    # Constant repetitions make the ordinary t statistic undefined. These two
    # limiting cases still provide useful, deterministic results to callers.
    numerical_scale = max(1.0, maximum(abs, float_values), abs(float_reference))
    tolerance = eps(Float64) * numerical_scale
    statistic, probability = if standard_error <= tolerance
        if abs(difference) <= tolerance
            (0.0, 1.0)
        else
            (copysign(Inf, difference), 0.0)
        end
    else
        t_statistic = difference / standard_error
        degrees_of_freedom = length(float_values) - 1
        two_sided_pvalue = 2 * ccdf(TDist(degrees_of_freedom), abs(t_statistic))
        (t_statistic, clamp(two_sided_pvalue, 0.0, 1.0))
    end

    (
        statistic=statistic,
        pvalue=probability,
        standard_error=standard_error,
        degrees_of_freedom=length(float_values) - 1,
    )
end

export ks_test, reference_value_test
