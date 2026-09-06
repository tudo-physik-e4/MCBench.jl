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

function _ess_filename(testcase, sampler)
    "$(testcase.info)-EffectiveSampleSize-$(sampler.info).txt"
end

function _ess_output_path(testcase, sampler)
    joinpath("teststatistics_sampler", _ess_filename(testcase, sampler))
end

function _matched_iid_size_filename(testcase, sampler)
    "$(testcase.info)-MatchedIIDSampleSize-$(sampler.info).txt"
end

function _matched_iid_size_output_path(testcase, sampler)
    joinpath(
        "teststatistics_sampler",
        _matched_iid_size_filename(testcase, sampler),
    )
end

_match_iid_to_sampler_ess(sampler, unweight::Bool, iid::Bool) =
    unweight && !iid && sampler isa SamplingAlgorithm &&
    !(sampler isa IIDSamplingAlgorithm)

"""
    build_teststatistic(testcase, metrics;
                        s=IIDSampler(), n=100, n_steps=100_000,
                        n_samples=100_000, par=true, clean=false,
                        unweight=true, use_sampler=true, iid=false,
                        ess_pilot_runs=3, verbose=false)

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
- `n_samples`: number of samples passed to each ordinary metric repetition.
  When positive, draws are concatenated until this size is available and
  unused samples are retained for the following repetition. In ESS-matched
  mode (a non-IID sampling algorithm with `unweight=true`), every complete
  sampler draw is retained instead; its size is controlled by `s` or
  `n_steps`, while pilot ESS estimates determine a fixed IID batch size.
- `par`: calculate different metrics concurrently for a repetition. Sampling
  and the `n` repetitions remain sequential. With one metric, this has no
  effect. Custom metrics used with `par=true` should not mutate their shared
  sample input or unsynchronized global state.
- `clean`: overwrite existing statistic files when `true`; append when
  `false`.
- `unweight`: enable ESS-matched IID comparison for a non-IID
  `SamplingAlgorithm` when `true`. The sampler draw remains intact, while an
  IID batch size is selected from the smallest autocorrelation-based ESS
  across dimensions in preliminary draws. Both sampler and matched IID
  statistics are written by the same call. Weighted draws from other sampler
  types are resampled using Kish ESS.
- `use_sampler`: select how each draw is configured. When `true`, call
  `sample(testcase, s)` so settings stored in `s` control the draw. When
  `false`, call `sample(testcase, s; n_steps=n_steps)` so this function's
  `n_steps` value is forwarded explicitly. The sampler `s` is used in both
  cases; this keyword does not turn sampling on or off.
- `iid`: route output to the IID `teststatistics/` directory and omit the
  sampler name from the filename, even when `s` is not an IID sampler. It does
  not change how samples are generated.
- `ess_pilot_runs`: number of independent sampler draws used to choose the
  fixed IID batch size in ESS-matched mode. MCBench takes the minimum ESS
  across dimensions for each pilot and uses the median of those minima. When
  appending to existing ESS-matched results, the stored IID size is reused and
  no new pilots are drawn.
- `verbose`: print occasional repetition progress when `true`.

# Output

IID results are stored as
`teststatistics/<testcase>-<metric>.txt`. Results from other samplers are
stored as
`teststatistics_sampler/<testcase>-<metric>-<sampler>.txt`, unless `iid=true`.
In ESS-matched mode the per-repetition ESS values are additionally stored in
`teststatistics_sampler/<testcase>-EffectiveSampleSize-<sampler>.txt`, and the
fixed IID size is stored alongside them as `<testcase>-MatchedIIDSampleSize-<sampler>.txt`.

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
    ess_pilot_runs::Int=3,
    verbose::Bool=false,
)
    isempty(metrics) && throw(ArgumentError("metrics cannot be empty"))
    n > 0 || throw(ArgumentError("n must be positive"))
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    ess_pilot_runs > 0 || throw(ArgumentError("ess_pilot_runs must be positive"))

    paths = _statistic_output_paths(testcase, metrics, s, iid)
    match_iid = _match_iid_to_sampler_ess(s, unweight, iid)
    iid_paths = match_iid ? _statistic_output_paths(
        testcase,
        metrics,
        IIDSampler(),
        false,
    ) : String[]
    ess_path = match_iid ? _ess_output_path(testcase, s) : nothing
    matched_iid_size_path = match_iid ?
        _matched_iid_size_output_path(testcase, s) : nothing

    if match_iid && !clean &&
       (!isfile(ess_path) || !isfile(matched_iid_size_path))
        existing_outputs = filter(
            isfile,
            vcat(paths, iid_paths, [ess_path, matched_iid_size_path]),
        )
        isempty(existing_outputs) || throw(ArgumentError(
            "existing statistic files are not marked as ESS-matched; " *
            "use clean=true to replace them",
        ))
    end

    mode = clean ? "w" : "a"
    streams = IO[]
    iid_streams = IO[]
    ess_stream = nothing

    try
        append!(streams, (open(path, mode) for path in paths))
        if match_iid
            append!(iid_streams, (open(path, mode) for path in iid_paths))
            ess_stream = open(ess_path, mode)
            _build_ess_matched_teststatistics(
                testcase,
                n,
                metrics,
                streams,
                iid_streams,
                ess_stream;
                s=s,
                n_steps=n_steps,
                ess_pilot_runs=ess_pilot_runs,
                matched_iid_size_path=matched_iid_size_path,
                clean=clean,
                par=par,
                use_sampler=use_sampler,
                verbose=verbose,
            )
        else
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
        end
    finally
        foreach(stream -> isopen(stream) && close(stream), streams)
        foreach(stream -> isopen(stream) && close(stream), iid_streams)
        !isnothing(ess_stream) && isopen(ess_stream) && close(ess_stream)
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

function _write_metric_result(
    io,
    testcase,
    samples,
    metric,
    sampler,
    two_sample_reference,
)
    result = if metric isa TwoSampleMetric && !isnothing(two_sample_reference)
        run_teststatistic(
            testcase,
            two_sample_reference,
            samples,
            metric,
            sampler,
        )
    else
        run_teststatistic(testcase, samples, metric, sampler)
    end
    println(io, JSON.json([value.val for value in result]))
end

function _write_metric_results(
    streams,
    testcase,
    samples,
    metrics,
    sampler,
    parallel::Bool,
    two_sample_reference=nothing,
)
    if parallel && length(metrics) > 1
        Folds.foreach(eachindex(metrics)) do index
            _write_metric_result(
                streams[index],
                testcase,
                samples,
                metrics[index],
                sampler,
                two_sample_reference,
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
                two_sample_reference,
            )
        end
    end
    nothing
end

function _validated_effective_sample_count(samples, sampler)
    effective_size = get_effective_sample_size(samples, sampler)
    isfinite(effective_size) && effective_size >= 1 || error(
        "sampler produced an invalid effective sample size: $effective_size",
    )
    floor(Int, effective_size)
end

function _build_ess_matched_teststatistics(
    testcase,
    n,
    metrics,
    sampler_streams,
    iid_streams,
    ess_stream;
    s,
    n_steps,
    ess_pilot_runs,
    matched_iid_size_path,
    clean,
    par,
    use_sampler,
    verbose,
)
    progress_interval = max(1, fld(n, 10))
    has_two_sample_metric = any(metric -> metric isa TwoSampleMetric, metrics)
    effective_sizes = Int[]
    sizehint!(effective_sizes, n)

    draw_sampler() = use_sampler ?
        sample(testcase, s) :
        sample(testcase, s; n_steps=n_steps)

    matched_iid_size = if !clean && isfile(matched_iid_size_path)
        read_matched_iid_sample_size(testcase, s)
    else
        pilot_sizes = Int[]
        sizehint!(pilot_sizes, ess_pilot_runs)
        for _ in 1:ess_pilot_runs
            pilot_samples = draw_sampler()
            isempty(pilot_samples) && error("sampler returned no pilot samples")
            push!(
                pilot_sizes,
                _validated_effective_sample_count(pilot_samples, s),
            )
        end
        selected_size = floor(Int, median(pilot_sizes))
        open(matched_iid_size_path, "w") do io
            println(io, JSON.json(selected_size))
        end
        verbose && println(
            "ESS pilots: $(pilot_sizes); matched IID sample size: $selected_size",
        )
        selected_size
    end

    for repetition in 1:n
        sampler_samples = draw_sampler()
        isempty(sampler_samples) && error("sampler returned no samples")

        effective_size = _validated_effective_sample_count(sampler_samples, s)
        push!(effective_sizes, effective_size)
        println(ess_stream, JSON.json(effective_size))

        # Keep IID draws independent between the sampler comparison and the
        # IID baseline. This avoids coupling their statistic distributions.
        sampler_reference = has_two_sample_metric ?
            sample(testcase; n_steps=matched_iid_size) : nothing
        iid_samples = sample(testcase; n_steps=matched_iid_size)
        iid_reference = has_two_sample_metric ?
            sample(testcase; n_steps=matched_iid_size) : nothing

        _write_metric_results(
            sampler_streams,
            testcase,
            sampler_samples,
            metrics,
            s,
            par,
            sampler_reference,
        )
        _write_metric_results(
            iid_streams,
            testcase,
            iid_samples,
            metrics,
            IIDSampler(),
            par,
            iid_reference,
        )

        if verbose && (repetition == 1 || repetition == n || repetition % progress_interval == 0)
            println(
                "Completed $repetition/$n ESS-matched repetitions " *
                "(current ESS: $effective_size; IID size: $matched_iid_size)",
            )
        end
    end

    if verbose
        println(
            "Sampler ESS values: " *
            "min=$(minimum(effective_sizes)), " *
            "median=$(median(effective_sizes)), " *
            "max=$(maximum(effective_sizes)); " *
            "fixed IID size=$matched_iid_size",
        )
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
4. If `n_samples > 0`, draws continue until the buffer contains exactly
   `n_samples` prepared samples. Surplus samples are retained for the next
   repetition. Otherwise, each complete prepared draw is one batch.
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
                isempty(buffer) && error("sampler returned no samples")
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


"""
    read_teststatistic(testcase, metric)
    read_teststatistic(testcase, metric, sampler)

Read persisted metric repetitions. The first form reads the IID baseline; the
second reads results for `sampler`. The returned matrix has one row per metric
output and one column per repetition.
"""
function read_teststatistic(testcase::AbstractTestcase, metric::TestMetric)
    parse_teststatistic(joinpath(
        "teststatistics",
        _statistic_filename(testcase, metric),
    ))
end

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

"""
    read_effective_sample_sizes(testcase, sampler)

Read the conservative autocorrelation-based ESS recorded for every sampler
repetition in an ESS-matched benchmark. Returns a vector of integer sample
counts.
"""
function read_effective_sample_sizes(
    testcase::AbstractTestcase,
    sampler::SamplingAlgorithm,
)
    path = _ess_output_path(testcase, sampler)
    values = open(path, "r") do io
        JSON.parse.(readlines(io))
    end
    Int.(values)
end

"""
    read_matched_iid_sample_size(testcase, sampler)

Read the fixed IID batch size chosen by the ESS pilot runs for this testcase
and sampler.
"""
function read_matched_iid_sample_size(
    testcase::AbstractTestcase,
    sampler::SamplingAlgorithm,
)
    path = _matched_iid_size_output_path(testcase, sampler)
    open(path, "r") do io
        Int(JSON.parse(readline(io)))
    end
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

export read_teststatistic, read_effective_sample_sizes
export read_matched_iid_sample_size
export parse_teststatistic, parseline
