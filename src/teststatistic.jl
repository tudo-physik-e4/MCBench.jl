"""Calculate a one-sample metric from an existing density sample vector."""
function run_teststatistic(
    testcase::AbstractTestcase,
    samples::DensitySampleVector,
    metric::TestMetric,
    ::AnySampler,
)
    calc_metric(testcase, samples, metric)
end

function run_teststatistic(
    testcase::AbstractTestcase,
    samples::DensitySampleVector,
    metric::TestMetric,
    marker::Int,
)
    metric isa TwoSampleMetric && return run_teststatistic_two_sample_metric(
        testcase,
        samples,
        metric,
        marker,
    )
    calc_metric(testcase, samples, metric)
end

export run_teststatistic


function _statistic_filename(testcase, metric, sampler=nothing)
    sampler_suffix = isnothing(sampler) ? "" : "-$(sampler.info)"
    "$(testcase.info)-$(metric.info)$(sampler_suffix).txt"
end

function _statistic_output_paths(testcase, metrics, sampler, iid::Bool)
    is_iid = sampler isa IIDSamplingAlgorithm || iid
    output_dir = is_iid ? "teststatistics" : "teststatistics_sampler"
    mkpath(output_dir)
    selected_sampler = is_iid ? nothing : sampler
    [
        joinpath(output_dir, _statistic_filename(testcase, metric, selected_sampler))
        for metric in metrics
    ]
end

"""
    build_teststatistic(testcase, metrics;
                        s=IIDSampler(), n=100, n_steps=100_000,
                        n_samples=100_000, par=true, clean=false,
                        unweight=true, use_sampler=true, iid=false,
                        verbose=false)

Build empirical test-statistic distributions for `metrics`.

For each of the `n` repetitions, this function obtains samples from `s`,
evaluates every metric on the same sample batch, and writes each result as one
JSON array per line. It returns the paths of the metric files it created or
appended to. Output directories are created automatically.

# Keywords

- `s`: sampler used to obtain candidate samples. The default `IIDSampler()`
  draws directly from the testcase.
- `n`: number of independent metric repetitions, and therefore the number of
  lines written to each output file.
- `n_steps`: sample count requested from each call to `sample` only when
  `use_sampler=false`.
- `n_samples`: number of samples passed to each metric repetition. When
  positive, draws are concatenated until this size is available and unused
  samples are retained for the following repetition. When non-positive, one
  complete sampler draw is used per repetition.
- `par`: calculate different metrics concurrently for a repetition. Sampling
  and the `n` repetitions remain sequential. With one metric, this has no
  effect. Custom metrics used with `par=true` should not mutate their shared
  sample input or unsynchronized global state.
- `clean`: overwrite existing statistic files when `true`; append when
  `false`.
- `unweight`: convert weighted draws to an unweighted sample of approximately
  their Kish effective sample size before batching when `true`.
- `use_sampler`: select how each draw is configured. When `true`, call
  `sample(testcase, s)` so settings stored in `s` control the draw. When
  `false`, call `sample(testcase, s; n_steps=n_steps)` so this function's
  `n_steps` value is forwarded explicitly. The sampler `s` is used in both
  cases; this keyword does not turn sampling on or off.
- `iid`: route output to the IID `teststatistics/` directory and omit the
  sampler name from the filename, even when `s` is not an IID sampler. It does
  not change how samples are generated.
- `verbose`: print occasional repetition progress when `true`.

# Output

IID results are stored as
`teststatistics/<testcase>-<metric>.txt`. Results from other samplers are
stored as
`teststatistics_sampler/<testcase>-<metric>-<sampler>.txt`, unless `iid=true`.

# Examples

Use the sampler's own configuration:

```julia
sampler = IIDSampler(5_000, "IID")
paths = build_teststatistic(testcase, metrics; s=sampler, n=20, clean=true)
```

Explicitly request 5,000 samples from every sampler draw and evaluate metrics
on batches of 1,000 samples:

```julia
paths = build_teststatistic(
    testcase,
    metrics;
    s=sampler,
    n=20,
    n_steps=5_000,
    n_samples=1_000,
    use_sampler=false,
    par=true,
    clean=true,
)
```
"""
function build_teststatistic(
    testcase::AbstractTestcase,
    metrics::AbstractVector{<:TestMetric};
    s=IIDSampler(),
    n::Int=100,
    n_steps::Int=100_000,
    n_samples::Int=100_000,
    par::Bool=true,
    clean::Bool=false,
    unweight::Bool=true,
    use_sampler::Bool=true,
    iid::Bool=false,
    verbose::Bool=false,
)
    isempty(metrics) && throw(ArgumentError("metrics cannot be empty"))
    n > 0 || throw(ArgumentError("n must be positive"))
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))

    paths = _statistic_output_paths(testcase, metrics, s, iid)
    mode = clean ? "w" : "a"
    streams = IO[]

    try
        append!(streams, (open(path, mode) for path in paths))
        build_teststat_reshuffle(
            testcase,
            n,
            metrics,
            streams;
            s=s,
            n_steps=n_steps,
            n_samples=n_samples,
            unweight=unweight,
            par=par,
            use_sampler=use_sampler,
            verbose=verbose,
        )
    finally
        foreach(stream -> isopen(stream) && close(stream), streams)
    end

    paths
end

export build_teststatistic


function _prepare_statistic_samples(samples, unweight::Bool)
    if unweight && is_weighted(samples)
        return resample_dsv_to_ess(samples)
    end
    samples
end

function _write_metric_result(io, testcase, samples, metric, sampler)
    result = run_teststatistic(testcase, samples, metric, sampler)
    println(io, JSON.json([value.val for value in result]))
end

function _write_metric_results(
    streams,
    testcase,
    samples,
    metrics,
    sampler,
    parallel::Bool,
)
    if parallel && length(metrics) > 1
        Folds.foreach(eachindex(metrics)) do index
            _write_metric_result(
                streams[index],
                testcase,
                samples,
                metrics[index],
                sampler,
            )
        end
    else
        for index in eachindex(metrics)
            _write_metric_result(
                streams[index],
                testcase,
                samples,
                metrics[index],
                sampler,
            )
        end
    end
    nothing
end

"""
    build_teststat_reshuffle(testcase, n, metrics, streams;
                             s=IIDSampler(), n_steps=100_000,
                             n_samples=0, unweight=true, par=false,
                             use_sampler=true, verbose=false)

Run the sampling, batching, and metric-evaluation loop used by
[`build_teststatistic`](@ref).

`streams` must contain one writable stream for every entry in `metrics`, in
the same order. One result is written to each stream for every repetition.
Streams are owned by the caller: this function neither opens nor closes them.
It returns `nothing`.

Sampling and batching follow these rules:

1. With `use_sampler=true`, a draw is obtained with `sample(testcase, s)`, so
   the sampler's stored/default configuration is used.
2. With `use_sampler=false`, the draw uses
   `sample(testcase, s; n_steps=n_steps)`.
3. If `unweight=true`, weighted draws are resampled to approximately their
   Kish effective sample size.
4. If `n_samples > 0`, draws are buffered until a batch of exactly
   `n_samples` is available. Otherwise, each complete draw is one batch.
5. Every metric is evaluated on the same batch. `par=true` evaluates those
   metrics concurrently; it does not parallelize sampler draws or repetitions.

Most callers should use [`build_teststatistic`](@ref), which validates the
public inputs, creates the output files, and guarantees stream cleanup.
"""
function build_teststat_reshuffle(
    testcase::AbstractTestcase,
    n::Int,
    metrics::AbstractVector{<:TestMetric},
    streams::AbstractVector{<:IO};
    s=IIDSampler(),
    n_steps::Int=100_000,
    n_samples::Int=0,
    unweight::Bool=true,
    par::Bool=false,
    use_sampler::Bool=true,
    verbose::Bool=false,
)
    length(streams) == length(metrics) || throw(DimensionMismatch(
        "one output stream is required per metric",
    ))

    draw_samples() = _prepare_statistic_samples(
        use_sampler ? sample(testcase, s) : sample(testcase, s; n_steps=n_steps),
        unweight,
    )

    progress_interval = max(1, fld(n, 10))
    buffer = nothing

    for repetition in 1:n
        samples = if n_samples <= 0
            draw_samples()
        else
            if isnothing(buffer)
                buffer = draw_samples()
            end
            while length(buffer) < n_samples
                additional = draw_samples()
                isempty(additional) && error("sampler returned no samples")
                buffer = vcat(buffer, additional)
            end

            selected = buffer[1:n_samples]
            buffer = length(buffer) == n_samples ? buffer[1:0] : buffer[n_samples + 1:end]
            selected
        end

        _write_metric_results(streams, testcase, samples, metrics, s, par)
        if verbose && (repetition == 1 || repetition == n || repetition % progress_interval == 0)
            println("Completed $repetition/$n test-statistic repetitions")
        end
    end

    nothing
end


"""Read persisted IID statistics for one testcase and metric."""
function read_teststatistic(testcase::AbstractTestcase, metric::TestMetric)
    parse_teststatistic(joinpath(
        "teststatistics",
        _statistic_filename(testcase, metric),
    ))
end

"""Read persisted sampler statistics for one testcase and metric."""
function read_teststatistic(
    testcase::AbstractTestcase,
    metric::TestMetric,
    sampler::AnySampler,
)
    parse_teststatistic(joinpath(
        "teststatistics_sampler",
        _statistic_filename(testcase, metric, sampler),
    ))
end

"""Parse line-delimited JSON statistic vectors into a dimensions-by-runs matrix."""
function parse_teststatistic(filename::AbstractString)
    rows = open(filename, "r") do io
        JSON.parse.(readlines(io))
    end
    isempty(rows) && throw(ArgumentError("test-statistic file is empty: $filename"))

    row_length = length(first(rows))
    all(row -> length(row) == row_length, rows) || throw(DimensionMismatch(
        "test-statistic rows have inconsistent lengths in: $filename",
    ))
    reshape(Float64.(vcat(rows...)), row_length, length(rows))
end

"""Parse one JSON statistic vector from an open stream."""
function parseline(io::IO)
    line = readline(io)
    isempty(line) ? Float64[] : Float64.(JSON.parse(line))
end

export read_teststatistic, parse_teststatistic, parseline
