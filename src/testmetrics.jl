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

"""Marginal sample mean for every testcase dimension."""
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
    [marginal_mean(value) for value in _as_metric_vector(BAT.mean(samples))]
end

export marginal_mean


"""Marginal sample variance for every testcase dimension."""
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
    [marginal_variance(value) for value in _as_metric_vector(BAT.var(samples))]
end

export marginal_variance


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


"""Weighted marginal skewness for every testcase dimension."""
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


"""Sliced Wasserstein distance over random one-dimensional projections."""
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


"""Gaussian-kernel maximum mean discrepancy between two samples."""
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
