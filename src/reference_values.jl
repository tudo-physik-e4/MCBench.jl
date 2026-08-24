"""
    reference_values(testcase, metric)

Return the known population values attached to `testcase` for `metric`, or
`nothing` when no reference is available.

Reference keys match metric type names. For example, `marginal_mean()` reads
the `:marginal_mean` entry. A scalar is broadcast to the metric's output
dimension; vectors must already have the expected length. Configurable metrics
may store a function that accepts the metric and returns the corresponding
scalar or vector.
"""
function reference_values(testcase::AbstractTestcase, metric::TestMetric)
    hasproperty(testcase, :reference_values) || return nothing

    stored_values = getproperty(testcase, :reference_values)
    key = nameof(typeof(metric))
    hasproperty(stored_values, key) || return nothing

    expected_length = metric_output_dimension(testcase, metric)
    stored_value = getproperty(stored_values, key)
    raw_value = stored_value isa Function ? stored_value(metric) : stored_value
    isnothing(raw_value) && return nothing
    raw_value isa Real || raw_value isa AbstractVector{<:Real} || throw(ArgumentError(
        "reference function :$key must return a real number, a real vector, or nothing",
    ))
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

metric_output_dimension(
    testcase::AbstractTestcase,
    metric::marginal_quantiles,
) = testcase.dim * length(metric.probabilities)

# These metrics summarize a complete multivariate sample with one scalar.
metric_output_dimension(
    ::AbstractTestcase,
    ::Union{sliced_wasserstein_distance,maximum_mean_discrepancy},
) = 1

"""Return human-readable labels for every value produced by a metric."""
function metric_output_labels(
    testcase::AbstractTestcase,
    metric::TestMetric;
    names::AbstractVector=String[],
)
    if !isempty(names) && length(names) != testcase.dim
        throw(DimensionMismatch("names must contain one label per testcase dimension"))
    end

    dimensions = metric_output_dimension(testcase, metric)
    metric isa Union{sliced_wasserstein_distance,maximum_mean_discrepancy} &&
        return [string(metric.info)]
    parameter_names = isempty(names) ? ["x$dim" for dim in 1:testcase.dim] : names
    ["$(metric.info)($(parameter_names[dim]))" for dim in 1:dimensions]
end

function metric_output_labels(
    testcase::AbstractTestcase,
    metric::marginal_quantiles;
    names::AbstractVector=String[],
)
    if !isempty(names) && length(names) != testcase.dim
        throw(DimensionMismatch("names must contain one label per testcase dimension"))
    end

    parameter_names = isempty(names) ? ["x$dim" for dim in 1:testcase.dim] : names
    [
        "$(_quantile_percentage(probability))% quantile ($(parameter_names[dim]))"
        for probability in metric.probabilities
        for dim in 1:testcase.dim
    ]
end


export reference_values, metric_output_dimension, metric_output_labels
