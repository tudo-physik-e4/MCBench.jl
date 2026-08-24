function _validate_metric_summary_inputs(testcase, metrics, names, comparison)
    isempty(metrics) && throw(ArgumentError("metrics cannot be empty"))
    comparison in (:iid, :reference) || throw(ArgumentError(
        "comparison must be :iid or :reference",
    ))
    if !isempty(names) && length(names) != testcase.dim
        throw(DimensionMismatch("names must contain one label per testcase dimension"))
    end
end

function _standardized_difference(difference, scale)
    isfinite(scale) && !iszero(scale) ? difference / scale : nothing
end

"""
    metric_summary_rows(testcase, metrics, sampler;
                        names=[], comparison=:iid)

Return the numeric rows used by [`metric_summary`](@ref). Each row contains the
metric label, sampler mean and standard deviation, comparison value and
optional standard deviation, raw difference, and standardized difference.

With `comparison=:iid`, all selected metrics are compared with their empirical
IID distributions. The standardized difference uses `std(IID metric)`. With
`comparison=:reference`, metrics without stored reference values are skipped,
and the standardized difference uses the sampler standard deviation so it can
be read against the reference plot's sigma bands.
"""
function metric_summary_rows(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    names::AbstractVector=String[],
    comparison::Symbol=:iid,
)
    _validate_metric_summary_inputs(testcase, metrics, names, comparison)
    rows = NamedTuple[]

    for metric in metrics
        references = comparison === :reference ? reference_values(testcase, metric) : nothing
        comparison === :reference && isnothing(references) && continue

        sampler_values = read_teststatistic(testcase, metric, sampler)
        dimensions = _validate_statistic_dimensions(testcase, metric, sampler_values)
        labels = metric_output_labels(testcase, metric; names=names)

        iid_values = if comparison === :iid
            values = read_teststatistic(testcase, metric)
            _validate_statistic_dimensions(testcase, metric, values)
            values
        else
            nothing
        end

        for dim in 1:dimensions
            sampler_row = sampler_values[dim, :]
            sampler_mean = mean(sampler_row)
            sampler_std = std(sampler_row)

            comparison_mean, comparison_std = if comparison === :iid
                iid_row = iid_values[dim, :]
                (mean(iid_row), std(iid_row))
            else
                (references[dim], nothing)
            end

            difference = sampler_mean - comparison_mean
            scale = comparison === :iid ? comparison_std : sampler_std
            push!(rows, (
                metric=labels[dim],
                sampler_mean=sampler_mean,
                sampler_std=sampler_std,
                comparison=comparison,
                comparison_mean=comparison_mean,
                comparison_std=comparison_std,
                difference=difference,
                standardized_difference=_standardized_difference(difference, scale),
            ))
        end
    end

    isempty(rows) && throw(ArgumentError(
        "none of the selected metrics has a reference value for testcase $(testcase.info)",
    ))
    rows
end

function _summary_number(value, digits)
    isnothing(value) && return "—"
    isfinite(value) || return string(value)
    string(round(value; sigdigits=digits))
end

function _summary_pad(value, width; align_right=false)
    padding = repeat(" ", max(0, width - textwidth(value)))
    align_right ? padding * value : value * padding
end

function _format_metric_summary_table(rows, comparison, digits)
    headers = if comparison === :iid
        ["Metric", "Sampler mean", "Sampler σ", "IID mean", "IID σ", "Δ", "Δ / σ_IID"]
    else
        ["Metric", "Sampler mean", "Sampler σ", "Reference", "Δ", "Δ / σ_sampler"]
    end

    cells = [
        comparison === :iid ?
        [
            string(row.metric),
            _summary_number(row.sampler_mean, digits),
            _summary_number(row.sampler_std, digits),
            _summary_number(row.comparison_mean, digits),
            _summary_number(row.comparison_std, digits),
            _summary_number(row.difference, digits),
            _summary_number(row.standardized_difference, digits),
        ] :
        [
            string(row.metric),
            _summary_number(row.sampler_mean, digits),
            _summary_number(row.sampler_std, digits),
            _summary_number(row.comparison_mean, digits),
            _summary_number(row.difference, digits),
            _summary_number(row.standardized_difference, digits),
        ]
        for row in rows
    ]

    widths = [
        maximum(textwidth(row[column]) for row in [[headers]; cells])
        for column in eachindex(headers)
    ]
    format_row(row) = join(
        [
            _summary_pad(row[column], widths[column]; align_right=column > 1)
            for column in eachindex(row)
        ],
        " | ",
    )
    separator = join([repeat("-", width) for width in widths], "-+-")
    join([format_row(headers), separator, format_row.(cells)...], "\n")
end

"""
    metric_summary(testcase, metrics, sampler;
                   names=[], comparison=:iid, digits=6)

Create a text table summarizing persisted sampler metric results.

The sampler result is reported as its empirical mean and standard deviation
across benchmark repetitions. Use `comparison=:iid` for the empirical IID mean
and standard deviation, or `comparison=:reference` for exact testcase reference
values. Reference mode includes only metrics for which a reference is stored.
The returned string is not printed automatically.
"""
function metric_summary(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    names::AbstractVector=String[],
    comparison::Symbol=:iid,
    digits::Int=6,
)
    digits > 0 || throw(ArgumentError("digits must be positive"))
    rows = metric_summary_rows(
        testcase,
        metrics,
        sampler;
        names=names,
        comparison=comparison,
    )
    heading = "$(testcase.info) — $(sampler.info) — $(uppercase(string(comparison))) comparison"
    "$heading\n$(_format_metric_summary_table(rows, comparison, digits))"
end

"""Print [`metric_summary`](@ref) to `io` and return the generated string."""
function print_metric_summary(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    io::IO=stdout,
    kwargs...,
)
    summary = metric_summary(testcase, metrics, sampler; kwargs...)
    println(io, summary)
    summary
end

export metric_summary_rows, metric_summary, print_metric_summary
