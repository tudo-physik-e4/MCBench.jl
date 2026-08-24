"""
    reference_values(testcase, metric)

Return the known population values attached to `testcase` for `metric`, or
`nothing` when no reference is available.

Reference keys match metric type names. For example, `marginal_mean()` reads
the `:marginal_mean` entry. A scalar is broadcast to the metric's output
dimension; vectors must already have the expected length.
"""
function reference_values(testcase::AbstractTestcase, metric::TestMetric)
    hasproperty(testcase, :reference_values) || return nothing

    stored_values = getproperty(testcase, :reference_values)
    key = nameof(typeof(metric))
    hasproperty(stored_values, key) || return nothing

    expected_length = metric_output_dimension(testcase, metric)
    raw_value = getproperty(stored_values, key)
    values = raw_value isa Real ?
        fill(Float64(raw_value), expected_length) :
        Float64.(collect(raw_value))

    length(values) == expected_length || throw(DimensionMismatch(
        "reference value :$key has length $(length(values)); expected $expected_length",
    ))

    values
end

"""Return the number of values produced by a metric for a testcase."""
metric_output_dimension(testcase::AbstractTestcase, ::TestMetric) = testcase.dim

# These metrics summarize a complete multivariate sample with one scalar.
metric_output_dimension(
    ::AbstractTestcase,
    ::Union{sliced_wasserstein_distance,maximum_mean_discrepancy},
) = 1

export reference_values, metric_output_dimension
