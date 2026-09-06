"""
    one_sample_ks_test(values, distribution; effective_sample_size=nothing)

Compare finite scalar observations with a continuous analytic distribution
using HypothesisTests.jl's asymptotic, two-sided one-sample Kolmogorov-Smirnov
test. Returns a named tuple with `statistic` and `pvalue`.

When `effective_sample_size` is supplied, the KS statistic is still calculated
from all observations, but its p-value is calibrated with
`floor(effective_sample_size)`. This provides an approximate correction for
autocorrelated samples. The effective sample size must lie between one and the
number of observations.

The p-value assumes independent observations and a fully specified continuous
reference distribution. Replacing the raw count with an autocorrelation-based
ESS is an approximation to this assumption, not a general dependent-sample KS
test. For a discrete reference, provide an appropriate custom
`goodness_of_fit` function to [`AnalyticReferenceDistribution`](@ref).
"""
function one_sample_ks_test(
    values::AbstractVector{<:Real},
    distribution::ContinuousUnivariateDistribution,
    ;
    effective_sample_size=nothing,
)
    isempty(values) && throw(ArgumentError(
        "analytic-reference values cannot be empty",
    ))
    all(isfinite, values) || throw(ArgumentError(
        "analytic-reference values must be finite",
    ))

    test = ApproximateOneSampleKSTest(Float64.(values), distribution)
    effective_count = _validated_ks_effective_sample_size(
        effective_sample_size,
        length(values),
    )
    calibrated_test = effective_count == length(values) ? test :
        ApproximateOneSampleKSTest(effective_count, test.δ, test.δp, test.δn)
    (
        statistic=Float64(test.δ),
        pvalue=Float64(pvalue(calibrated_test; tail=:both)),
    )
end

function _validated_ks_effective_sample_size(effective_sample_size, sample_count)
    isnothing(effective_sample_size) && return sample_count
    effective_sample_size isa Real && !(effective_sample_size isa Bool) ||
        throw(ArgumentError("effective_sample_size must be a real number"))
    isfinite(effective_sample_size) || throw(ArgumentError(
        "effective_sample_size must be finite",
    ))
    1 <= effective_sample_size <= sample_count || throw(ArgumentError(
        "effective_sample_size must lie between one and the number of observations",
    ))
    floor(Int, effective_sample_size)
end

function one_sample_ks_test(
    ::AbstractVector{<:Real},
    ::UnivariateDistribution,
    ;
    effective_sample_size=nothing,
)
    throw(ArgumentError(
        "the default one-sample KS p-value is only distribution-free for a " *
        "continuous reference; provide a discrete goodness_of_fit function",
    ))
end

export one_sample_ks_test


"""
    reference_distribution(testcase, name)

Return the [`AnalyticReferenceDistribution`](@ref) stored under `name`, or
`nothing` when the testcase has no such reference. This metadata is separate
from both scalar [`reference_values`](@ref) and empirically generated IID
test-statistic distributions.
"""
function reference_distribution(
    testcase::AbstractTestcase,
    name::Union{Symbol,AbstractString},
)
    hasproperty(testcase, :reference_distributions) || return nothing
    references = getproperty(testcase, :reference_distributions)
    key = Symbol(name)
    hasproperty(references, key) ? getproperty(references, key) : nothing
end

export reference_distribution


"""Result of comparing transformed samples with an analytic distribution."""
struct ReferenceDistributionTestResult{R,V<:AbstractVector{Float64},A}
    testcase_info::A
    name::Symbol
    reference::R
    observable_values::V
    statistic::Float64
    pvalue::Float64
    effective_sample_size::Int
end

# Preserve the original constructor for downstream code that creates results
# directly. Without an explicit ESS, every observation is treated as effective.
function ReferenceDistributionTestResult(
    testcase_info,
    name,
    reference,
    observable_values::AbstractVector{Float64},
    statistic,
    pvalue,
)
    ReferenceDistributionTestResult(
        testcase_info,
        name,
        reference,
        observable_values,
        Float64(statistic),
        Float64(pvalue),
        length(observable_values),
    )
end

function Base.show(io::IO, result::ReferenceDistributionTestResult)
    print(
        io,
        "ReferenceDistributionTestResult(",
        result.testcase_info,
        ", ",
        result.reference.info,
        "; ",
        result.reference.test_name,
        ": statistic=",
        result.statistic,
        ", pvalue=",
        result.pvalue,
        ", n=",
        length(result.observable_values),
        result.effective_sample_size == length(result.observable_values) ? "" :
            ", n_eff=$(result.effective_sample_size)",
        ")",
    )
end

export ReferenceDistributionTestResult


function _validated_analytic_test_result(result, reference)
    hasproperty(result, :statistic) && hasproperty(result, :pvalue) || throw(ArgumentError(
        "$(reference.info) goodness_of_fit must return statistic and pvalue fields",
    ))
    statistic = getproperty(result, :statistic)
    probability = getproperty(result, :pvalue)
    statistic isa Real && isfinite(statistic) || throw(ArgumentError(
        "$(reference.info) goodness-of-fit statistic must be finite",
    ))
    probability isa Real && isfinite(probability) && 0 <= probability <= 1 ||
        throw(ArgumentError(
            "$(reference.info) goodness-of-fit p-value must be finite and between zero and one",
        ))
    (statistic=Float64(statistic), pvalue=Float64(probability))
end

function _analytic_observable_values(
    testcase::AbstractTestcase,
    samples::DensitySampleVector,
    reference::AnalyticReferenceDistribution;
    unweight::Bool,
)
    isempty(samples) && throw(ArgumentError("samples cannot be empty"))
    prepared_samples = if is_weighted(samples)
        unweight || throw(ArgumentError(
            "one-sample goodness-of-fit tests require unweighted observations; " *
            "set unweight=true to resample weighted input",
        ))
        resample_dsv_to_ess(samples)
    else
        samples
    end
    isempty(prepared_samples) && throw(ArgumentError(
        "weighted samples have an effective sample size below one",
    ))

    coordinates = _sample_value_matrix(prepared_samples)
    size(coordinates, 1) == testcase.dim || throw(DimensionMismatch(
        "samples have $(size(coordinates, 1)) dimensions; expected $(testcase.dim)",
    ))

    transformed = Vector{Float64}(undef, size(coordinates, 2))
    for index in axes(coordinates, 2)
        sample_value = view(coordinates, :, index)
        applicable(reference.observable, sample_value) || throw(ArgumentError(
            "$(reference.info) observable must accept one sample coordinate vector",
        ))
        value = reference.observable(sample_value)
        value isa Real && isfinite(value) || throw(ArgumentError(
            "$(reference.info) observable must return one finite real number per sample",
        ))
        transformed[index] = Float64(value)
    end
    transformed
end

"""
    reference_distribution_test(testcase, samples, name;
                                unweight=true,
                                effective_sample_size=nothing)
    reference_distribution_test(testcase, sampler, name;
                                n_steps=100_000, use_sampler=true,
                                unweight=true, correct_for_ess=true)

Apply the observation-level transform stored in `testcase.reference_distributions`
under `name`, then compare the resulting scalar observations directly with its
analytic distribution. No IID reference sample is generated.

Weighted `DensitySampleVector` input is resampled to its Kish effective sample
size by default. Set `unweight=false` to reject weighted input instead. The
default KS p-value uses every resulting observation unless
`effective_sample_size` is supplied. For sampler-based calls,
`correct_for_ess=true` automatically uses the sampler-aware ESS; for MCMC this
is the smallest autocorrelation-based ESS across dimensions. The KS statistic
continues to use the full empirical CDF, while only its p-value calibration is
changed. This is an approximate ESS correction rather than an exact test for
dependent observations.
"""
function reference_distribution_test(
    testcase::AbstractTestcase,
    samples::DensitySampleVector,
    name::Union{Symbol,AbstractString};
    unweight::Bool=true,
    effective_sample_size=nothing,
)
    key = Symbol(name)
    reference = reference_distribution(testcase, key)
    isnothing(reference) && throw(ArgumentError(
        "testcase $(testcase.info) has no analytic reference distribution :$key",
    ))

    values = _analytic_observable_values(
        testcase,
        samples,
        reference;
        unweight=unweight,
    )
    effective_count = _validated_ks_effective_sample_size(
        effective_sample_size,
        length(values),
    )
    raw_result = if effective_count == length(values)
        reference.goodness_of_fit(values, reference.distribution)
    elseif reference.goodness_of_fit === one_sample_ks_test
        one_sample_ks_test(
            values,
            reference.distribution;
            effective_sample_size=effective_count,
        )
    elseif applicable(
        reference.goodness_of_fit,
        values,
        reference.distribution,
        effective_count,
    )
        reference.goodness_of_fit(
            values,
            reference.distribution,
            effective_count,
        )
    else
        throw(ArgumentError(
            "$(reference.info) uses a custom goodness_of_fit function that " *
            "does not accept effective_sample_size as a third argument",
        ))
    end
    result = _validated_analytic_test_result(raw_result, reference)
    ReferenceDistributionTestResult(
        testcase.info,
        key,
        reference,
        values,
        result.statistic,
        result.pvalue,
        effective_count,
    )
end

function reference_distribution_test(
    testcase::AbstractTestcase,
    values::AbstractMatrix{<:Real},
    name::Union{Symbol,AbstractString};
    kwargs...,
)
    reference_distribution_test(testcase, make_dsv(values), name; kwargs...)
end

function reference_distribution_test(
    testcase::AbstractTestcase,
    values::AbstractVector{<:Real},
    name::Union{Symbol,AbstractString};
    kwargs...,
)
    testcase.dim == 1 || throw(DimensionMismatch(
        "a vector of sample values is only accepted for a one-dimensional testcase",
    ))
    reference_distribution_test(testcase, make_dsv(values), name; kwargs...)
end

function reference_distribution_test(
    testcase::AbstractTestcase,
    sampler::AnySampler,
    name::Union{Symbol,AbstractString};
    n_steps::Int=100_000,
    use_sampler::Bool=true,
    unweight::Bool=true,
    correct_for_ess::Bool=true,
)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    samples = use_sampler ?
        sample(testcase, sampler) :
        sample(testcase, sampler; n_steps=n_steps)
    effective_sample_size = if correct_for_ess && sampler isa SamplingAlgorithm
        min(
            get_effective_sample_size(samples, sampler),
            floor(Int, get_effective_sample_size(samples)),
        )
    else
        nothing
    end
    reference_distribution_test(
        testcase,
        samples,
        name;
        unweight=unweight,
        effective_sample_size=effective_sample_size,
    )
end

export reference_distribution_test


"""
    mahalanobis_reference(distribution)

Construct the analytic squared-Mahalanobis reference for a univariate or
multivariate normal distribution. For `X ~ N_d(μ, Σ)`, the stored observable
is `(X-μ)'Σ⁻¹(X-μ)` and its reference distribution is `Chisq(d)`.
"""
function mahalanobis_reference(distribution::Normal)
    location = Float64(mean(distribution))
    scale_squared = Float64(var(distribution))
    observable = sample_value ->
        abs2(Float64(only(sample_value)) - location) / scale_squared
    AnalyticReferenceDistribution(
        observable,
        Chisq(1);
        info="Squared Mahalanobis distance",
    )
end


function mahalanobis_reference(distribution::AbstractMvNormal)
    location = Float64.(mean(distribution))
    covariance_factor = cholesky(Symmetric(Matrix{Float64}(cov(distribution))))
    observable = sample_value -> begin
        difference = Float64.(sample_value) .- location
        sum(abs2, covariance_factor.L \ difference)
    end
    AnalyticReferenceDistribution(
        observable,
        Chisq(length(location));
        info="Squared Mahalanobis distance",
    )
end

export mahalanobis_reference


function _reference_plot_title(result::ReferenceDistributionTestResult)
    statistic = round(result.statistic; sigdigits=5)
    probability = round(result.pvalue; sigdigits=5)
    sample_count = length(result.observable_values)
    statistic_name = result.reference.goodness_of_fit === one_sample_ks_test ?
        "D" : "statistic"
    test_result = "$(result.reference.test_name): " *
        "$statistic_name = $statistic, p = $probability"
    ess_note = result.effective_sample_size == sample_count ? "" :
        "\nESS-adjusted: n_eff = $(result.effective_sample_size)"
    "$(result.reference.info)\n$test_result$ess_note"
end

function _reference_sample_label(result::ReferenceDistributionTestResult)
    sample_count = length(result.observable_values)
    result.effective_sample_size == sample_count && return "Empirical, n = $sample_count"
    "Empirical, n = $sample_count, ESS = $(result.effective_sample_size)"
end

function _continuous_distribution_panel(result, nbins, legend_position)
    values = result.observable_values
    distribution = result.reference.distribution
    lower_reference = quantile(distribution, 0.001)
    upper_reference = quantile(distribution, 0.999)
    lower = min(
        minimum(values),
        isfinite(lower_reference) ? lower_reference : minimum(values),
    )
    upper = max(
        maximum(values),
        isfinite(upper_reference) ? upper_reference : maximum(values),
    )
    if lower == upper
        lower -= 0.5
        upper += 0.5
    end
    grid = range(lower, upper; length=500)

    panel = histogram(
        values;
        bins=nbins,
        normalize=:pdf,
        alpha=0.35,
        label=_reference_sample_label(result),
        xlabel=result.reference.info,
        ylabel="Density",
        legend=legend_position,
    )
    plot!(
        panel,
        grid,
        pdf.(Ref(distribution), grid);
        color=:black,
        linewidth=2,
        label="Analytic $(distribution)",
    )
    panel
end

function _discrete_reference_points(distribution, values)
    points = sort(unique(values))
    lower = quantile(distribution, 0.001)
    upper = quantile(distribution, 0.999)
    if isfinite(lower) && isfinite(upper) && isinteger(lower) && isinteger(upper) &&
       upper - lower <= 2_000
        points = sort(unique(vcat(points, collect(lower:upper))))
    end
    points
end

function _discrete_distribution_panel(result, legend_position)
    values = result.observable_values
    distribution = result.reference.distribution
    points = _discrete_reference_points(distribution, values)
    empirical = [count(==(point), values) / length(values) for point in points]

    panel = bar(
        points,
        empirical;
        alpha=0.35,
        label=_reference_sample_label(result),
        xlabel=result.reference.info,
        ylabel="Probability",
        legend=legend_position,
    )
    scatter!(
        panel,
        points,
        pdf.(Ref(distribution), points);
        markershape=:diamond,
        color=:black,
        label="Analytic $(distribution)",
    )
    panel
end

function _normalized_panel_heights(panel_heights)
    panel_heights isa Union{Tuple,AbstractVector} || throw(ArgumentError(
        "panel_heights must be a two-element tuple or vector",
    ))
    length(panel_heights) == 2 || throw(ArgumentError(
        "panel_heights must contain the distribution and CDF heights",
    ))
    all(height -> height isa Real && !(height isa Bool), panel_heights) ||
        throw(ArgumentError("panel_heights must contain real numbers"))
    heights = Float64.(panel_heights)
    all(height -> isfinite(height) && height > 0, heights) ||
        throw(ArgumentError("panel_heights must be positive and finite"))
    heights ./ sum(heights)
end


"""
    plot_reference_distribution(result;
                                nbins=32,
                                title=nothing,
                                distribution_legend=:topright,
                                cdf_legend=:topleft,
                                panel_heights=(3, 1),
                                save_plots=true)
    plot_reference_distribution(testcase, samples, name; ...)

Plot transformed sample observations against their analytic reference. The
upper panel compares the empirical density or mass with the analytic law; the
lower panel compares the empirical and analytic CDFs used by a KS test. The
default 3:1 `panel_heights` ratio leaves the CDF compact while keeping it
available for interpreting the KS statistic.

By default, `title=nothing` generates a title containing the test statistic
and p-value. Supply a string to replace it, or `""` to hide it. The
`distribution_legend` and `cdf_legend` keywords independently control the
legend positions of the upper and lower panels using ordinary Plots.jl legend
positions.

With `save_plots=true`, one PDF is written below `<testcase.info>/` and its
path is returned. Set `save_plots=false` to return the unsaved plot object.
"""
function plot_reference_distribution(
    result::ReferenceDistributionTestResult;
    nbins::Int=32,
    title::Union{Nothing,AbstractString}=nothing,
    distribution_legend=:topright,
    cdf_legend=:topleft,
    panel_heights=(3, 1),
    save_plots::Bool=true,
)
    nbins > 0 || throw(ArgumentError("nbins must be positive"))
    normalized_heights = _normalized_panel_heights(panel_heights)
    resolved_title = isnothing(title) ? _reference_plot_title(result) : title
    distribution = result.reference.distribution
    distribution_panel = if distribution isa ContinuousUnivariateDistribution
        _continuous_distribution_panel(result, nbins, distribution_legend)
    else
        _discrete_distribution_panel(result, distribution_legend)
    end

    sorted_values = sort(result.observable_values)
    empirical_cdf = collect(eachindex(sorted_values)) ./ length(sorted_values)
    cdf_panel = plot(
        sorted_values,
        empirical_cdf;
        st=:steppost,
        color=:steelblue,
        linewidth=2,
        label="Empirical CDF",
        xlabel=result.reference.info,
        ylabel="CDF",
        ylims=(0, 1),
        legend=cdf_legend,
    )
    plot!(
        cdf_panel,
        sorted_values,
        cdf.(Ref(distribution), sorted_values);
        color=:black,
        linewidth=2,
        label="Analytic CDF",
    )

    combined_plot = plot(
        distribution_panel,
        cdf_panel;
        layout=grid(2, 1; heights=normalized_heights),
        size=(650, 650),
        title=[resolved_title ""],
        bottom_margin=3Plots.mm,
    )
    save_plots || return combined_plot

    output_dir = string(result.testcase_info)
    mkpath(output_dir)
    output_path = joinpath(
        output_dir,
        "$(result.testcase_info)-$(result.name)-analytic-reference.pdf",
    )
    savefig(combined_plot, output_path)
    output_path
end

function plot_reference_distribution(
    testcase::AbstractTestcase,
    samples::Union{
        DensitySampleVector,
        AbstractMatrix{<:Real},
        AbstractVector{<:Real},
    },
    name::Union{Symbol,AbstractString};
    unweight::Bool=true,
    effective_sample_size=nothing,
    nbins::Int=32,
    title::Union{Nothing,AbstractString}=nothing,
    distribution_legend=:topright,
    cdf_legend=:topleft,
    panel_heights=(3, 1),
    save_plots::Bool=true,
)
    result = reference_distribution_test(
        testcase,
        samples,
        name;
        unweight=unweight,
        effective_sample_size=effective_sample_size,
    )
    plot_reference_distribution(
        result;
        nbins=nbins,
        title=title,
        distribution_legend=distribution_legend,
        cdf_legend=cdf_legend,
        panel_heights=panel_heights,
        save_plots=save_plots,
    )
end

function plot_reference_distribution(
    testcase::AbstractTestcase,
    sampler::AnySampler,
    name::Union{Symbol,AbstractString};
    n_steps::Int=100_000,
    use_sampler::Bool=true,
    unweight::Bool=true,
    correct_for_ess::Bool=true,
    nbins::Int=32,
    title::Union{Nothing,AbstractString}=nothing,
    distribution_legend=:topright,
    cdf_legend=:topleft,
    panel_heights=(3, 1),
    save_plots::Bool=true,
)
    result = reference_distribution_test(
        testcase,
        sampler,
        name;
        n_steps=n_steps,
        use_sampler=use_sampler,
        unweight=unweight,
        correct_for_ess=correct_for_ess,
    )
    plot_reference_distribution(
        result;
        nbins=nbins,
        title=title,
        distribution_legend=distribution_legend,
        cdf_legend=cdf_legend,
        panel_heights=panel_heights,
        save_plots=save_plots,
    )
end

export plot_reference_distribution
