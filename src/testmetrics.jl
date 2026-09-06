"""
    abstract type TestMetric

Base type for benchmark metrics. Concrete metrics store a calculated `val` and
an `info` label and implement an appropriate `calc_metric` method.
"""
abstract type TestMetric end

"""Base type for metrics that compare two sample clouds."""
abstract type TwoSampleMetric <: TestMetric end

export TestMetric, TwoSampleMetric


function _as_metric_vector(value::NamedTuple)
    _as_metric_vector(first(values(value)))
end

_as_metric_vector(value::Real) = [Float64(value)]
_as_metric_vector(value::AbstractVector) = Float64.(value)

"""
    marginal_mean()

Metric for the weighted sample mean of each testcase dimension. It produces
one output per dimension, in the same order as the sample coordinates.
"""
struct marginal_mean{V<:Real,A} <: TestMetric
    val::V
    info::A
end

marginal_mean() = marginal_mean(0.0, "Mean")
marginal_mean(value::Real) = marginal_mean(value, "Mean")

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    ::marginal_mean,
)
    values = _sample_value_matrix(samples)
    weights = FrequencyWeights(samples.weight)
    [marginal_mean(mean(row, weights)) for row in eachrow(values)]
end

export marginal_mean


"""
    marginal_variance()

Metric for the corrected weighted sample variance of each testcase dimension.
It produces one output per dimension.
"""
struct marginal_variance{V<:Real,A} <: TestMetric
    val::V
    info::A
end

marginal_variance() = marginal_variance(0.0, "Variance")
marginal_variance(value::Real) = marginal_variance(value, "Variance")

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    ::marginal_variance,
)
    values = _sample_value_matrix(samples)
    weights = FrequencyWeights(samples.weight)
    [marginal_variance(var(row, weights; corrected=true)) for row in eachrow(values)]
end

export marginal_variance


const _DEFAULT_QUANTILE_PROBABILITIES = (0.5, 0.9, 0.99)

function _validate_quantile_probabilities(probabilities)
    values = Tuple(Float64.(probabilities))
    isempty(values) && throw(ArgumentError("at least one quantile is required"))
    all(value -> isfinite(value) && 0 <= value <= 1, values) || throw(ArgumentError(
        "quantile probabilities must be finite values between zero and one",
    ))
    length(unique(values)) == length(values) || throw(ArgumentError(
        "quantile probabilities must be unique",
    ))
    values
end

function _quantile_percentage(probability)
    percentage = 100 * probability
    isinteger(percentage) ? string(Int(percentage)) : string(round(percentage; sigdigits=8))
end

function _quantile_info(probabilities)
    prefix = length(probabilities) == 1 ? "Quantile" : "Quantiles"
    percentages = join(_quantile_percentage.(probabilities), "-")
    "$prefix-$percentages"
end

"""
    marginal_quantiles()
    marginal_quantiles(probabilities)
    marginal_quantiles(; percent=nothing, percentages=nothing)

Weighted marginal sample quantiles for every testcase dimension. With no
arguments, calculate the 50%, 90%, and 99% quantiles. Positional values use
probabilities between zero and one:

```julia
marginal_quantiles(0.95)
marginal_quantiles([0.25, 0.5, 0.75])
```

For percentage notation, use `percent=95` or `percentages=[25, 50, 75]`.
Results are ordered by quantile and then by testcase dimension.
"""
struct marginal_quantiles{V<:Real,P<:Tuple,A} <: TestMetric
    val::V
    probabilities::P
    info::A
end

function marginal_quantiles(probabilities::Union{Real,Tuple,AbstractVector})
    values = probabilities isa Real ? (probabilities,) : probabilities
    validated = _validate_quantile_probabilities(values)
    marginal_quantiles(0.0, validated, _quantile_info(validated))
end

function marginal_quantiles(; percent=nothing, percentages=nothing)
    !isnothing(percent) && !isnothing(percentages) && throw(ArgumentError(
        "provide either percent or percentages, not both",
    ))

    if isnothing(percent) && isnothing(percentages)
        return marginal_quantiles(_DEFAULT_QUANTILE_PROBABILITIES)
    end

    selected = isnothing(percentages) ? percent : percentages
    values = selected isa Real ? (selected,) : selected
    all(value -> value isa Real && isfinite(value) && 0 <= value <= 100, values) ||
        throw(ArgumentError("quantile percentages must lie between zero and 100"))
    marginal_quantiles(Float64.(collect(values)) ./ 100)
end

"""Convenience constructor for a single marginal quantile."""
function marginal_quantile(
    probability::Union{Nothing,Real}=nothing;
    percent::Union{Nothing,Real}=nothing,
)
    !isnothing(probability) && !isnothing(percent) && throw(ArgumentError(
        "provide either a probability or percent, not both",
    ))
    !isnothing(percent) && return marginal_quantiles(percent=percent)
    marginal_quantiles(isnothing(probability) ? 0.5 : probability)
end

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    metric::marginal_quantiles,
)
    values = _sample_value_matrix(samples)
    weights = FrequencyWeights(samples.weight)
    [
        marginal_quantiles(
            quantile(row, weights, probability),
            metric.probabilities,
            metric.info,
        )
        for probability in metric.probabilities
        for row in eachrow(values)
    ]
end

export marginal_quantile, marginal_quantiles


"""Coordinates of the highest-density sampled point."""
struct global_mode{V<:Real,A} <: TestMetric
    val::V
    info::A
end

global_mode() = global_mode(0.0, "Globalmode")
global_mode(value::Real) = global_mode(value, "Globalmode")

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    ::global_mode,
)
    [global_mode(value) for value in _as_metric_vector(BAT.mode(samples))]
end

export global_mode


"""BAT's independently estimated marginal mode for every dimension."""
struct marginal_mode{V<:Real,A} <: TestMetric
    val::V
    info::A
end

marginal_mode() = marginal_mode(0.0, "Marginalmode")
marginal_mode(value::Real) = marginal_mode(value, "Marginalmode")

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    ::marginal_mode,
)
    result = BAT.bat_marginalmode(samples).result
    [marginal_mode(value) for value in _as_metric_vector(result)]
end

export marginal_mode


"""
    marginal_skewness()

Metric for the weighted marginal skewness of each testcase dimension. A
symmetric target has population skewness zero when that moment exists.
"""
struct marginal_skewness{V<:Real,A} <: TestMetric
    val::V
    info::A
end

marginal_skewness() = marginal_skewness(0.0, "Skewness")
marginal_skewness(value::Real) = marginal_skewness(value, "Skewness")

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    ::marginal_skewness,
)
    values = _sample_value_matrix(samples)
    weights = FrequencyWeights(samples.weight)
    [marginal_skewness(skewness(row, weights)) for row in eachrow(values)]
end

export marginal_skewness


"""Weighted marginal excess kurtosis for every testcase dimension."""
struct marginal_kurtosis{V<:Real,A} <: TestMetric
    val::V
    info::A
end

marginal_kurtosis() = marginal_kurtosis(0.0, "Kurtosis")
marginal_kurtosis(value::Real) = marginal_kurtosis(value, "Kurtosis")

function calc_metric(
    ::AbstractTestcase,
    samples::DensitySampleVector,
    ::marginal_kurtosis,
)
    values = _sample_value_matrix(samples)
    weights = FrequencyWeights(samples.weight)
    [marginal_kurtosis(kurtosis(row, weights)) for row in eachrow(values)]
end

export marginal_kurtosis


"""One-dimensional Wasserstein distance for each sample dimension."""
struct wasserstein_1d{V<:Real,A} <: TwoSampleMetric
    val::V
    info::A
end

wasserstein_1d() = wasserstein_1d(0.0, "Wasserstein")
wasserstein_1d(value::Real) = wasserstein_1d(value, "Wasserstein")

function calc_metric(
    testcase::AbstractTestcase,
    first::DensitySampleVector,
    second::DensitySampleVector,
    ::wasserstein_1d,
)
    first_values = _sample_value_matrix(first)
    second_values = _sample_value_matrix(second)

    [
        wasserstein_1d(wasserstein1d(
            Float64.(first_values[dim, :]),
            Float64.(second_values[dim, :]);
            wa=first.weight,
            wb=second.weight,
        ))
        for dim in 1:testcase.dim
    ]
end

export wasserstein_1d


"""
    sliced_wasserstein_distance()
    sliced_wasserstein_distance(0.0, max_samples)

Two-sample metric that averages one-dimensional Wasserstein distances over
random projections of the sample clouds. The default compares at most 100,000
observations from each input. The second form sets that limit explicitly;
`max_samples=0` uses all available observations.

The metric produces one value for the complete multivariate sample rather
than one value per dimension.
"""
struct sliced_wasserstein_distance{V<:Real,I<:Int,A,P} <: TwoSampleMetric
    val::V
    N::I
    info::A
    proc::P
end

wd_pint() = ""

sliced_wasserstein_distance() =
    sliced_wasserstein_distance(0.0, 100_000, "SlicedWasserstein", wd_pint())
sliced_wasserstein_distance(value::Real) =
    sliced_wasserstein_distance(value, 100_000, "SlicedWasserstein", "")
sliced_wasserstein_distance(value::Real, n::Int) =
    sliced_wasserstein_distance(value, n, "SlicedWasserstein", wd_pint())

function calc_metric(
    ::AbstractTestcase,
    first::DensitySampleVector,
    second::DensitySampleVector,
    metric::sliced_wasserstein_distance,
)
    [sliced_wasserstein_distance(get_sliced_wasserstein_distance(
        first,
        second;
        N=metric.N,
    ))]
end


export sliced_wasserstein_distance


"""
    maximum_mean_discrepancy()
    maximum_mean_discrepancy(0.0, max_samples)

Two-sample metric for Gaussian-kernel maximum mean discrepancy (MMD). The
default compares at most 10,000 observations from each input. The second form
sets that limit explicitly; `max_samples=0` uses all available observations.
The Gaussian-kernel scale is selected with the median-distance heuristic.

The metric produces one value for the complete multivariate sample.
"""
struct maximum_mean_discrepancy{V<:Real,I<:Int,A,P} <: TwoSampleMetric
    val::V
    N::I
    info::A
    proc::P
end

maximum_mean_discrepancy() =
    maximum_mean_discrepancy(0.0, 10_000, "MaximumMeanDiscrepancy", wd_pint())
maximum_mean_discrepancy(value::Real) =
    maximum_mean_discrepancy(value, 10_000, "MaximumMeanDiscrepancy", "")
maximum_mean_discrepancy(value::Real, n::Int) =
    maximum_mean_discrepancy(value, n, "MaximumMeanDiscrepancy", wd_pint())

function calc_metric(
    ::AbstractTestcase,
    first::DensitySampleVector,
    second::DensitySampleVector,
    metric::maximum_mean_discrepancy,
)
    [maximum_mean_discrepancy(get_mmd(first, second; N=metric.N))]
end

export maximum_mean_discrepancy


"""Legacy standardized two-sample discrepancy for each dimension."""
struct chi_squared{V<:Real,A} <: TwoSampleMetric
    val::V
    info::A
end

chi_squared() = chi_squared(0.0, "Chi-Squared")
chi_squared(value::Real) = chi_squared(value, "Chi-Squared")

function calc_metric(
    ::AbstractTestcase,
    first::DensitySampleVector,
    second::DensitySampleVector,
    ::chi_squared,
)
    chisq_test(first, second)
end

function chisq_test(first::DensitySampleVector, second::DensitySampleVector)
    first, second = prepare_twosample_dsv(first, second)
    first_values = _sample_value_matrix(first)
    second_values = _sample_value_matrix(second)

    first_means = mean.(eachrow(first_values))
    first_variances = var.(eachrow(first_values))
    [
        chi_squared(sum(second_values[dim, :] .- first_means[dim]) / first_variances[dim])
        for dim in axes(first_values, 1)
    ]
end

export chi_squared, calc_metric
