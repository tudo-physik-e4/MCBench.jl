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
                        names=[], comparison=:iid, include_reference=false)

Return the numeric rows used by [`metric_summary`](@ref). Each row contains the
metric label, sampler mean, standard deviation and standard error, comparison
value, raw difference, and standardized difference. IID comparison rows
additionally contain `ks_statistic` and `ks_pvalue` from a two-sided,
approximate two-sample Kolmogorov-Smirnov test.

With `comparison=:iid`, all selected metrics are compared with their empirical
IID distributions. Set `include_reference=true` to additionally attach each
available testcase reference as `reference_value`; metrics without an
analytical reference receive `nothing`. The standardized difference uses
`std(IID metric)`. With `comparison=:reference`, metrics without stored
reference values are skipped. Their standardized difference is the one-sample
t statistic `(sampler mean - reference) / sampler SEM`. Whenever a reference
is present, `reference_statistic` and `reference_pvalue` contain that t
statistic and its two-sided p-value.
"""
function metric_summary_rows(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    names::AbstractVector=String[],
    comparison::Symbol=:iid,
    include_reference::Bool=false,
)
    _validate_metric_summary_inputs(testcase, metrics, names, comparison)
    rows = NamedTuple[]

    for metric in metrics
        references = if comparison === :reference || include_reference
            reference_values(testcase, metric)
        else
            nothing
        end
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
            reference = isnothing(references) ? nothing : references[dim]
            comparison === :reference && ismissing(reference) && continue

            sampler_row = sampler_values[dim, :]
            sampler_mean = mean(sampler_row)
            sampler_std = std(sampler_row)
            sampler_sem = sampler_std / sqrt(length(sampler_row))

            comparison_mean, comparison_std = if comparison === :iid
                iid_row = iid_values[dim, :]
                (mean(iid_row), std(iid_row))
            else
                (reference, nothing)
            end

            difference = sampler_mean - comparison_mean
            ks_result = comparison === :iid ? ks_test(iid_row, sampler_row) : nothing
            reference_result = if isnothing(reference) || ismissing(reference)
                nothing
            else
                reference_value_test(sampler_row, reference)
            end
            standardized_difference = if comparison === :iid
                _standardized_difference(difference, comparison_std)
            else
                reference_result.statistic
            end
            push!(rows, (
                metric=labels[dim],
                sampler_mean=sampler_mean,
                sampler_std=sampler_std,
                sampler_sem=sampler_sem,
                comparison=comparison,
                comparison_mean=comparison_mean,
                comparison_std=comparison_std,
                reference_value=reference,
                difference=difference,
                standardized_difference=standardized_difference,
                ks_statistic=isnothing(ks_result) ? nothing : ks_result.statistic,
                ks_pvalue=isnothing(ks_result) ? nothing : ks_result.pvalue,
                reference_statistic=isnothing(reference_result) ? nothing : reference_result.statistic,
                reference_pvalue=isnothing(reference_result) ? nothing : reference_result.pvalue,
            ))
        end
    end

    isempty(rows) && throw(ArgumentError(
        "none of the selected metrics has a reference value for testcase $(testcase.info)",
    ))
    rows
end

function _summary_number(value, digits)
    (isnothing(value) || ismissing(value)) && return "—"
    isfinite(value) || return string(value)
    string(round(value; sigdigits=digits))
end

function _summary_pad(value, width; align_right=false)
    padding = repeat(" ", max(0, width - textwidth(value)))
    align_right ? padding * value : value * padding
end

function _format_metric_summary_table(rows, comparison, digits, include_reference)
    headers = if comparison === :iid
        [
            "Metric",
            "Sampler mean",
            "Sampler σ",
            "IID mean",
            "IID σ",
            "Δ vs IID",
            "Δ / σ_IID",
            "KS D",
            "KS p-value",
        ]
    else
        [
            "Metric",
            "Sampler mean",
            "Sampler σ",
            "Sampler SEM",
            "Reference",
            "Δ",
            "Δ / SEM",
            "Reference p-value",
        ]
    end

    if comparison === :iid && include_reference
        insert!(headers, 6, "Reference")
        insert!(headers, 7, "Reference p-value")
    end

    cells = Vector{Vector{String}}()
    for row in rows
        row_cells = if comparison === :iid
            [
                string(row.metric),
                _summary_number(row.sampler_mean, digits),
                _summary_number(row.sampler_std, digits),
                _summary_number(row.comparison_mean, digits),
                _summary_number(row.comparison_std, digits),
                _summary_number(row.difference, digits),
                _summary_number(row.standardized_difference, digits),
                _summary_number(row.ks_statistic, digits),
                _summary_number(row.ks_pvalue, digits),
            ]
        else
            [
                string(row.metric),
                _summary_number(row.sampler_mean, digits),
                _summary_number(row.sampler_std, digits),
                _summary_number(row.sampler_sem, digits),
                _summary_number(row.comparison_mean, digits),
                _summary_number(row.difference, digits),
                _summary_number(row.standardized_difference, digits),
                _summary_number(row.reference_pvalue, digits),
            ]
        end
        if comparison === :iid && include_reference
            insert!(row_cells, 6, _summary_number(row.reference_value, digits))
            insert!(row_cells, 7, _summary_number(row.reference_pvalue, digits))
        end
        push!(cells, row_cells)
    end

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
                   names=[], comparison=:iid, include_reference=false,
                   digits=6)

Create a text table summarizing persisted sampler metric results.

The sampler result is reported as its empirical mean and standard deviation
across benchmark repetitions. Use `comparison=:iid` for the empirical IID mean
and standard deviation, or `comparison=:reference` for exact testcase reference
values. In IID mode, `include_reference=true` adds a reference column while
retaining every selected metric; a dash marks metrics without an analytical
reference. IID mode also reports the KS statistic and its unadjusted p-value
for the complete distributions of repeated metric values. Whenever a reference
is shown, its p-value comes from a two-sided one-sample t-test across the
sampler repetitions. Reference mode includes only metrics for which a reference
is stored and reports both the sampler SEM and `(mean - reference) / SEM`.
At least two repetitions are needed for a reference test. All p-values are
reported without a multiple-testing correction. The returned string is not
printed automatically.
"""
function metric_summary(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    names::AbstractVector=String[],
    comparison::Symbol=:iid,
    include_reference::Bool=false,
    digits::Int=6,
)
    digits > 0 || throw(ArgumentError("digits must be positive"))
    rows = metric_summary_rows(
        testcase,
        metrics,
        sampler;
        names=names,
        comparison=comparison,
        include_reference=include_reference,
    )
    heading = "$(testcase.info) — $(sampler.info) — $(uppercase(string(comparison))) comparison"
    "$heading\n$(_format_metric_summary_table(rows, comparison, digits, include_reference))"
end

"""
    print_metric_summary(testcase, metrics, sampler; io=stdout, kwargs...)

Create a table with [`metric_summary`](@ref), print it to `io`, and return the
same string. All remaining keywords, including `comparison`, `names`,
`include_reference`, and `digits`, are forwarded to `metric_summary`.
"""
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
