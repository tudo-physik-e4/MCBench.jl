const _DEFAULT_PLOT_SIZE = (500, round(Int, 2 * 500 / 3))

function _metric_title(testcase, metric, dim)
    label = metric_output_labels(testcase, metric)[dim]
    "$(testcase.info)-$label"
end

function _validate_statistic_dimensions(testcase, metric, values)
    expected = metric_output_dimension(testcase, metric)
    size(values, 1) == expected || throw(DimensionMismatch(
        "stored $(metric.info) statistics have $(size(values, 1)) rows; expected $expected",
    ))
    expected
end

function _add_reference_line!(plot_object, testcase, metric, dim; show_reference::Bool)
    show_reference || return nothing
    references = reference_values(testcase, metric)
    isnothing(references) && return nothing

    value = references[dim]
    ismissing(value) && return nothing
    vline!(
        plot_object,
        [value];
        color=:black,
        linestyle=:dash,
        linewidth=2,
        label="Reference = $(round(value; sigdigits=5))",
    )
    nothing
end

function _metric_reference(testcase, metric, dim)
    references = reference_values(testcase, metric)
    isnothing(references) && return nothing
    value = references[dim]
    ismissing(value) ? nothing : value
end

function _ks_plot_label(result)
    statistic = round(result.statistic; sigdigits=4)
    probability = round(result.pvalue; sigdigits=4)
    "KS test: D = $statistic, p = $probability"
end

function _reference_plot_label(values, reference; source="")
    result = reference_value_test(values, reference)
    statistic = round(result.statistic; sigdigits=4)
    probability = round(result.pvalue; sigdigits=4)
    source_suffix = isempty(source) ? "" : " ($source)"
    "Reference t-test$source_suffix: t = $statistic, p = $probability"
end

"""
    plot_teststatistic(testcase, metric;
                       nbins=32, show_reference=true,
                       show_reference_test=true, save_plots=true)

Plot the IID test-statistic distribution for each metric output dimension. If
the testcase contains a matching reference value, it is drawn as a dashed
vertical line. Returns the paths of the generated PDF files.

When a reference is available, the title also shows the two-sided one-sample
t-test p-value for the repeated IID metric values. Disable this annotation with
`show_reference_test=false`.

Set `save_plots=false` to return the plot objects without writing files. This
is useful when callers want to customize or explicitly save each plot.
"""
function plot_teststatistic(
    testcase::AbstractTestcase,
    metric::TestMetric;
    nbins::Int=32,
    show_reference::Bool=true,
    show_reference_test::Bool=true,
    save_plots::Bool=true,
)
    values = read_teststatistic(testcase, metric)
    dimensions = _validate_statistic_dimensions(testcase, metric, values)
    output_dir = "teststatistics"
    save_plots && mkpath(output_dir)
    output_paths = String[]
    plot_objects = Plots.Plot[]

    for dim in 1:dimensions
        title = _metric_title(testcase, metric, dim)
        reference = _metric_reference(testcase, metric, dim)
        if show_reference_test && !isnothing(reference)
            title *= "\n" * _reference_plot_label(values[dim, :], reference; source="IID")
        end
        metric_plot = histogram(
            values[dim, :];
            bins=nbins,
            st=:stephist,
            title=title,
            xlabel=metric.info,
            ylabel="Entries",
            label="IID, n = $(size(values, 2))",
            size=_DEFAULT_PLOT_SIZE,
        )
        _add_reference_line!(
            metric_plot,
            testcase,
            metric,
            dim;
            show_reference=show_reference,
        )

        output_path = joinpath(
            output_dir,
            "$(testcase.info)-$(metric.info)-x$dim.pdf",
        )
        if save_plots
            savefig(metric_plot, output_path)
            push!(output_paths, output_path)
        else
            push!(plot_objects, metric_plot)
        end
    end

    save_plots ? output_paths : plot_objects
end


"""
    plot_teststatistic(testcase, metric, sampler;
                       nbins=32, same_bins=true, sampler_bins=false,
                       show_reference=true, show_reference_test=true,
                       show_ks_test=true,
                       save_plots=true)

Compare IID and sampler test-statistic distributions. Reference values stored
on the testcase are shown by default. The title reports the two-sample KS
statistic and p-value calculated from the unbinned values; disable this with
`show_ks_test=false`. If a stored reference is available, the title separately
reports a one-sample t-test of the sampler repetitions against it; disable this
with `show_reference_test=false`. Returns the generated PDF paths. With
`save_plots=false`, returns the unsaved plot objects instead.
"""
function plot_teststatistic(
    testcase::AbstractTestcase,
    metric::TestMetric,
    sampler::AnySampler;
    nbins::Int=32,
    same_bins::Bool=true,
    sampler_bins::Bool=false,
    show_reference::Bool=true,
    show_reference_test::Bool=true,
    show_ks_test::Bool=true,
    save_plots::Bool=true,
)
    iid_values = read_teststatistic(testcase, metric)
    sampler_values = read_teststatistic(testcase, metric, sampler)
    dimensions = _validate_statistic_dimensions(testcase, metric, iid_values)
    _validate_statistic_dimensions(testcase, metric, sampler_values)

    output_dir = string(testcase.info)
    save_plots && mkpath(output_dir)
    output_paths = String[]
    plot_objects = Plots.Plot[]

    for dim in 1:dimensions
        iid_row = Float64.(iid_values[dim, :])
        sampler_row = Float64.(sampler_values[dim, :])

        iid_histogram = if sampler_bins
            sampler_edges = fit(Histogram, sampler_row; nbins=nbins).edges[1]
            fit(Histogram, iid_row, sampler_edges)
        else
            fit(Histogram, iid_row; nbins=nbins)
        end

        sampler_histogram = if same_bins
            fit(Histogram, sampler_row, iid_histogram.edges[1])
        else
            fit(Histogram, sampler_row; nbins=nbins)
        end

        iid_histogram = normalize(iid_histogram)
        sampler_histogram = normalize(sampler_histogram)
        title = _metric_title(testcase, metric, dim)
        if show_ks_test
            title *= "\n$(_ks_plot_label(ks_test(iid_row, sampler_row)))"
        end
        reference = _metric_reference(testcase, metric, dim)
        if show_reference_test && !isnothing(reference)
            title *= "\n$(_reference_plot_label(sampler_row, reference; source=string(sampler.info)))"
        end

        metric_plot = plot(
            iid_histogram;
            st=:step,
            title=title,
            xlabel=metric.info,
            ylabel="Entries",
            label="IID, n = $(length(iid_row))",
            size=_DEFAULT_PLOT_SIZE,
        )
        plot!(
            metric_plot,
            sampler_histogram;
            st=:step,
            label="$(sampler.info), n = $(length(sampler_row))",
        )
        _add_reference_line!(
            metric_plot,
            testcase,
            metric,
            dim;
            show_reference=show_reference,
        )

        output_path = joinpath(
            output_dir,
            "$(testcase.info)-$(metric.info)-$(sampler.info)-x$dim.pdf",
        )
        if save_plots
            savefig(metric_plot, output_path)
            push!(output_paths, output_path)
        else
            push!(plot_objects, metric_plot)
        end
    end

    save_plots ? output_paths : plot_objects
end

export plot_teststatistic


function _has_value(value)
    !(isnothing(value) || (value isa AbstractArray && isempty(value)))
end

function _draw_std_band!(plot_object, lower, upper, color, plot_height, alpha)
    plot!(
        plot_object,
        [lower, upper],
        fill(2 * plot_height, 2);
        fillrange=fill(-plot_height, 2),
        label="",
        color=color,
        fillalpha=alpha,
        lw=0,
    )
end

"""
    plot_metrics(testcase, normalized_values; ...)

Render a normalized overview plot from rows containing `name`, `val`, and
optionally `std`. This lower-level method is also used by the metric-based
overload below. Returns the generated output paths, or the unsaved plot object
when `save_plots=false`.
"""
function plot_metrics(
    testcase::AbstractTestcase,
    normalized_values::AbstractVector{<:NamedTuple};
    plotalpha::Real=0.2,
    infos=nothing,
    s=nothing,
    xlabel="(metric - mean(IID metric)) / std(IID metric)",
    save_plots::Bool=true,
)
    isempty(normalized_values) && throw(ArgumentError("normalized_values cannot be empty"))

    rows = reverse(normalized_values)
    plot_height = length(rows) * 30
    y_values = [10 + 30 * (index - 1) for index in eachindex(rows)]

    overview_plot = plot(
        (0, 0);
        size=(550, plot_height + 100),
        legend=:topleft,
        xlims=(-3, 3),
        label="",
        bottom_margin=3Plots.mm,
        left_margin=10Plots.mm,
        framestyle=:box,
        dpi=400,
    )
    xlabel!(overview_plot, xlabel)

    max_x = 3.0
    for (row, y) in zip(rows, y_values)
        if haskey(row, :std)
            scatter!(overview_plot, (row.val, y); xerr=row.std, label="", color=:black)
            row_extent = abs(row.val) + abs(row.std)
        else
            scatter!(overview_plot, (row.val, y); label="", color=:black)
            row_extent = abs(row.val)
        end
        max_x = max(max_x, row_extent)
    end

    # Background bands provide a quick visual interpretation in IID standard
    # deviations without changing the metric-specific scale.
    vline!(overview_plot, [0]; label="", color=:black)
    _draw_std_band!(overview_plot, -1, 1, :green, plot_height, plotalpha)
    _draw_std_band!(overview_plot, -2, -1, :yellow, plot_height, plotalpha)
    _draw_std_band!(overview_plot, 1, 2, :yellow, plot_height, plotalpha)
    _draw_std_band!(overview_plot, -3, -2, :red, plot_height, plotalpha)
    _draw_std_band!(overview_plot, 2, 3, :red, plot_height, plotalpha)

    yticks!(
        overview_plot,
        y_values,
        [row.name for row in rows];
        size=(550, plot_height),
        xlims=(-max_x - 0.1, max_x + 0.1),
        ylims=(-10, plot_height + 10),
    )

    info_suffix = _has_value(infos) ? " $(infos)" : ""
    sampler_suffix = _has_value(s) ? "-$(s.info)" : ""
    title!(
        overview_plot,
        "$(testcase.info)$(sampler_suffix)$(info_suffix)";
        size=(550, plot_height + 100),
    )

    save_plots || return overview_plot

    output_dir = string(testcase.info)
    mkpath(output_dir)
    stem = "$(testcase.info)$(sampler_suffix)-metrics"
    output_paths = [
        joinpath(output_dir, "$stem.pdf"),
        joinpath(output_dir, "$stem.png"),
    ]
    savefig(overview_plot, output_paths[1])
    savefig(overview_plot, output_paths[2])
    output_paths
end

"""
    plot_metrics(testcase, metrics, sampler;
                 names=[], save_plots=true)

Create the normalized metric overview for a sampler. Each point is

`(mean(sampler metric) - mean(IID metric)) / std(IID metric)`.

Its error bar is the sampler metric's standard deviation on the same normalized
scale. The colored background marks one, two, and three IID standard deviations.

By default, the overview is saved as PDF and PNG and their paths are returned.
Set `save_plots=false` to return the plot object without writing either file.
"""
function plot_metrics(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    names::AbstractVector=String[],
    save_plots::Bool=true,
)
    isempty(metrics) && throw(ArgumentError("metrics cannot be empty"))
    if !isempty(names) && length(names) != testcase.dim
        throw(DimensionMismatch("names must contain one label per testcase dimension"))
    end

    normalized_values = NamedTuple[]

    for metric in metrics
        iid_values = read_teststatistic(testcase, metric)
        sampler_values = read_teststatistic(testcase, metric, sampler)
        dimensions = _validate_statistic_dimensions(testcase, metric, iid_values)
        _validate_statistic_dimensions(testcase, metric, sampler_values)
        labels = metric_output_labels(testcase, metric; names=names)

        for dim in 1:dimensions
            iid_row = iid_values[dim, :]
            sampler_row = sampler_values[dim, :]
            iid_scale = std(iid_row)
            iszero(iid_scale) && throw(ArgumentError(
                "cannot normalize $(metric.info) dimension $dim because its IID standard deviation is zero",
            ))

            normalized_mean = (mean(sampler_row) - mean(iid_row)) / iid_scale
            normalized_std = std(sampler_row) / iid_scale

            push!(normalized_values, (
                name=labels[dim],
                val=normalized_mean,
                std=normalized_std,
            ))
        end
    end

    plot_metrics(
        testcase,
        normalized_values;
        s=sampler,
        save_plots=save_plots,
    )
end

function _plot_reference_metrics(
    testcase::AbstractTestcase,
    reference_values_to_plot::AbstractVector{<:NamedTuple},
    sampler::AnySampler;
    save_plots::Bool,
)
    rows = reverse(reference_values_to_plot)
    plot_height = length(rows) * 30
    y_values = [10 + 30 * (index - 1) for index in eachindex(rows)]
    max_x = maximum(abs(row.val) + row.error for row in rows)
    x_padding = iszero(max_x) ? 1.0 : 0.05 * max_x

    overview_plot = plot(
        (0, 0);
        size=(650, plot_height + 100),
        legend=:topleft,
        xlims=(-max_x - x_padding, max_x + x_padding),
        label="",
        bottom_margin=3Plots.mm,
        left_margin=16Plots.mm,
        framestyle=:box,
        dpi=400,
    )
    xlabel!(overview_plot, "(mean(metric) - reference value) / SEM(metric)")

    vline!(overview_plot, [0]; color=:black, linestyle=:dash, label="Reference")
    for (index, (row, y)) in enumerate(zip(rows, y_values))
        scatter!(
            overview_plot,
            [row.val],
            [y];
            xerror=[row.error],
            color=:black,
            markerstrokecolor=:black,
            label=index == 1 ? "Difference ± 1 SEM" : "",
        )
    end

    yticks!(
        overview_plot,
        y_values,
        ["$(row.name) (p = $(round(row.pvalue; sigdigits=4)))" for row in rows];
        ylims=(-10, plot_height + 10),
    )
    title!(overview_plot, "$(testcase.info)-$(sampler.info) reference values")

    save_plots || return overview_plot

    output_dir = string(testcase.info)
    mkpath(output_dir)
    stem = "$(testcase.info)-$(sampler.info)-reference-metrics"
    output_paths = [
        joinpath(output_dir, "$stem.pdf"),
        joinpath(output_dir, "$stem.png"),
    ]
    savefig(overview_plot, output_paths[1])
    savefig(overview_plot, output_paths[2])
    output_paths
end

"""
    plot_reference_metrics(testcase, metrics, sampler;
                           names=[], save_plots=true)

Plot the selected metrics for which the testcase defines reference values.
Each point is the reference difference normalized to the standard error of the
mean across benchmark repetitions:

`(mean(sampler metric) - reference value) / SEM(sampler metric)`.

Here `SEM = std(sampler metric) / sqrt(n_repetitions)`, so every horizontal
error bar spans ±1 in normalized SEM units. At least two repetitions and a
positive finite SEM are required. Each row label reports the two-sided
one-sample t-test p-value for agreement with its reference. These p-values are
not adjusted for the number of displayed metrics.

Metric outputs without reference values are skipped individually. An
`ArgumentError` is raised if none of the selected outputs has a reference
value. By default, the plot is saved as PDF and PNG; use `save_plots=false` to
return the plot object instead.
"""
function plot_reference_metrics(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric},
    sampler::AnySampler;
    names::AbstractVector=String[],
    save_plots::Bool=true,
)
    isempty(metrics) && throw(ArgumentError("metrics cannot be empty"))
    if !isempty(names) && length(names) != testcase.dim
        throw(DimensionMismatch("names must contain one label per testcase dimension"))
    end

    values_to_plot = NamedTuple[]
    for metric in metrics
        references = reference_values(testcase, metric)
        isnothing(references) && continue
        all(ismissing, references) && continue

        sampler_values = read_teststatistic(testcase, metric, sampler)
        dimensions = _validate_statistic_dimensions(testcase, metric, sampler_values)
        labels = metric_output_labels(testcase, metric; names=names)

        for dim in 1:dimensions
            ismissing(references[dim]) && continue
            sampler_row = sampler_values[dim, :]
            repetitions = length(sampler_row)
            repetitions > 1 || throw(ArgumentError(
                "at least two repetitions are required to estimate the standard error for $(labels[dim])",
            ))
            standard_error = std(sampler_row) / sqrt(repetitions)
            numerical_scale = max(1.0, maximum(abs, sampler_row))
            minimum_sem = eps(Float64) * numerical_scale
            isfinite(standard_error) && standard_error > minimum_sem || throw(ArgumentError(
                "cannot normalize $(labels[dim]) because its standard error is not positive, finite, and distinguishable from zero",
            ))
            push!(values_to_plot, (
                name=labels[dim],
                val=(mean(sampler_row) - references[dim]) / standard_error,
                error=1.0,
                pvalue=reference_value_test(sampler_row, references[dim]).pvalue,
            ))
        end
    end

    isempty(values_to_plot) && throw(ArgumentError(
        "none of the selected metrics has a reference value for testcase $(testcase.info)",
    ))

    _plot_reference_metrics(
        testcase,
        values_to_plot,
        sampler;
        save_plots=save_plots,
    )
end

export plot_metrics, plot_reference_metrics
