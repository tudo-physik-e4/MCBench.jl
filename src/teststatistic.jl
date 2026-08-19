"""
    run_teststatistic(t, samples::DensitySampleVector, m::TM, s)

Calculate and return a metric value based on the provided test case, sample vector, metric, and sampler.

# Arguments
- `t`: A test case. This can either be of `AbstractTestcase` subtype or simply a `Testcases` object.
- `samples::DensitySampleVector`: A vector containing the density samples.
- `m::TM`: The metric to be calculated, where `TM` is a subtype of `TestMetric`.
- `s`: A sampler or configuration. This can either be of type `AS` (any subtype of `AnySampler`) or an `Int`.

# Returns
- `TestMetric`: The calculated metric value.

# Notes
This function has two implementations to accommodate different secnatios. In case the metrics are calculated on IID samples an `Int` can be passed as the sampler as an actual sampler is not needed.  

"""
# Overloaded function to run a test statistic with a test case, density sample vector, and sampling algorithm
function run_teststatistic(t::AT, samples::DensitySampleVector, m::TM, s::AS) where {TM <: TestMetric, AT <: AbstractTestcase, AS <: AnySampler}
    if m isa TwoSampleMetric
        return run_teststatistic_two_sample_metric(t, samples, m, s)
    end
    mval = calc_metric(t, samples, m)
end

# Overloaded function to run a test statistic with a test case, density sample vector, and integer
function run_teststatistic(t::Testcases, samples::DensitySampleVector, m::TM, s::Int) where {TM <: TestMetric}
    mval = calc_metric(t, samples, m)
end
export run_teststatistic


"""
    build_teststatistic(t<:AbstractTestcase, m::Vector{TestMetric}; 
        s=IIDSampler(), n::Int=10^2, n_steps::Int=10^5, 
        n_samples::Int=10^5, par::Bool=true, 
        clean::Bool=false, unweight::Bool=true, 
        use_sampler::Bool=true, iid=false)

Build and store test statistics for a given test case and multiple Metrics. This function manages file creation and invokes reshuffling and sampling to calculate the desired statistics.
The resampling is performed `n` times according to the effective sample size of the provided sampler.
Sampling is repeted until `n samples` of effective samples are generated.

# Arguments
- `t<:AbstractTestcase`: The test case, is a subtype of `AbstractTestcase`.
- `m::Vector{TM}`: A vector of metrics, where `TM` is a subtype of `TestMetric`.

# Keyword Arguments
- `s`: A sampling algorithm. Defaults to `IIDSampler()`.
- `n::Int`: Number of iterations aka the number of times the tests and calculations of the metrics is performed. Defaults to `10^2`.
- `n_steps::Int`: Number of steps for the sampling algorithm. This setting is only necessary for samplers that require a fixed number of steps. This is not relevant if the number of steps is defined in the sampler object. Defaults to `10^5`. 
- `n_samples::Int`: Number of samples to generate. This setting referes to the number of IID samples. Defaults to `10^5`. 
- `par::Bool`: Whether to enable parallel processing. Defaults to `true`.
- `clean::Bool`: If `true`, clears the output files before calculation. Defaults to `false`.
- `unweight::Bool`: Whether to unweight samples during reshuffling. If `false` resampling to the effective sample size will be skipped. Defaults to `true`. 
- `use_sampler::Bool`: Whether to utilize the sampler in the calculations. This is only necessary for internal sampler types which have custom settings (like fixed number of steps). It will use the actual sampling object instead of creating a new instance of the sampler object with default settings and the number of steps in this function. Defaults to `true`.
- `iid::Bool`: If `true`, forces IID behavior regardless of the sampler type. Can be used for IID to IID comparisons. Defaults to `false`.

# Returns
- `Nothing`: Results are stored in specified files.

# Notes
1. Make sure to have created `teststatistics` and `teststatistics_sampler` filepaths.
2. File paths are generated based on the `info` property of the test case, metric, and sampler.
3. If `clean` is `true`, all files are overwritten. Otherwise, results are appended.
4. The function ensures that files are properly closed, even in case of errors.
5. Reshuffling and sampling are handled by the `build_teststat_reshuffle` function.

# Example
```julia
metrics = [Metric1(), Metric2()]
testcase = MyTestcase()
build_teststatistic(testcase, metrics; n=100, par=true, clean=true)
```
"""
function build_teststatistic(t::T, m::Vector{TM}; s=IIDSampler(), n::Int=10^2, n_steps::Int=10^5, n_samples::Int=10^5, par::Bool=true, clean::Bool=false, unweight::Bool=true, use_sampler::Bool=true, iid=false) where {T<:AbstractTestcase, TM <: TestMetric}
    fnm = [string("./teststatistics/", string(t.info, "-", im.info, ".txt")) for im in m]
    if !isa(s, IIDSamplingAlgorithm) && !iid
        fnm = [string("./teststatistics_sampler/", string(t.info, "-", im.info, "-", s.info, ".txt")) for im in m]
    end
    if clean
        fs = [open(f, "w+") for f in fnm]
        close.(fs)
    end

    fs = [open(f, "a+") for f in fnm]
    try
        build_teststat_reshuffle(t, n, m, fs, s=s, n_steps=n_steps, n_samples=n_samples, unweight=unweight, par=par, use_sampler=use_sampler)
    finally
        # `build_teststat_reshuffle` closes the streams after a successful run.
        # Closing an already closed stream is harmless, and the `finally` block
        # also guarantees cleanup without hiding exceptions from callers.
        close.(fs)
    end
end
export build_teststatistic

"""
    build_teststat_reshuffle(t<:AbstractTestcase, n::Int, m::Vector{TestMetric}, fnm::Vector{IOStream}; 
        s=IIDSampler(), n_steps=10^5, n_samples=0, unweight=true, par=false, use_sampler=true) -> Nothing

This function is prone to be changed mostly used for internal usage, please use the `build_teststatistic` function instead!
Generates reshuffled test statistics for a given test case and metric vector, writing results to specified files.

# Arguments
- `t<:AbstractTestcase`: The test case to be evaluated, is a subtype of `AbstractTestcase`.
- `n::Int`: Number of reshuffling iterations to perform.
- `m::Vector{TM}`: A vector of metrics to evaluate, where `TM` is a subtype of `TestMetric`.
- `fnm::Vector{IOStream}`: A vector of open file streams for writing test statistics.

# Keyword Arguments
- `s`: The sampling algorithm to use. Defaults to `IIDSampler()`.
- `n_steps::Int`: Number of steps for the sampling algorithm. Defaults to `10^5`.
- `n_samples::Int`: Number of samples to generate for each iteration. Defaults to `0` (use all available samples).
- `unweight::Bool`: Whether to resample based on effective sample size (ESS) when weights are unequal. Defaults to `true`.
- `par::Bool`: Enable parallel processing if `true`. Defaults to `false`.
- `use_sampler::Bool`: Use the sampler for data generation if `true`. Defaults to `true`.

# Returns
- `Nothing`: Results are written to the provided file streams.

# Notes
1. Handles both cases of generating fixed or variable sample sizes (`n_samples > 0` or `n_samples <= 0`).
2. Ensures proper resampling when weights are unequal, maintaining consistent ESS.
3. Supports parallel and sequential processing based on the `par` argument.
4. Automatically closes all file streams upon completion.

# Example
```julia
testcase = MyTestcase()
metrics = [Metric1(), Metric2()]
file_streams = [open("metric1.txt", "w"), open("metric2.txt", "w")]
build_teststat_reshuffle(testcase, 100, metrics, file_streams; n_steps=10^4, unweight=true, par=false)
```

# Error Handling
- Ensures file streams are closed in case of unexpected errors during execution.
"""
function build_teststat_reshuffle(t::T, n::Int, m::Vector{TM}, fnm::Vector{IOStream}; 
    s=IIDSampler(), n_steps=10^5, n_samples=0, unweight=true, par=false, use_sampler=true) where {TM <: TestMetric, T<:AbstractTestcase}
    draw_samples() = use_sampler ? sample(t, s) : sample(t, s, n_steps=n_steps)
    prepare_samples(v) = if unweight && v.weight != ones(length(v))
        resample_dsv_to_ess(v, s)
    else
        v
    end

    j = 0
    while j < n
        v = prepare_samples(draw_samples())
        if n_samples <= 0
            for i in 1:length(m)
                write(fnm[i], string([j.val for j in run_teststatistic(t, v, m[i], s)], "\n"))
            end
            progress_every = max(1, fld(n_steps, 100))
            j % progress_every == 0 && println(j)
            j += 1
            continue
        end
        if n_samples > 0
            nv = v
            while j < n
                while length(nv) < n_samples
                    rv = prepare_samples(draw_samples())
                    nv = vcat(nv, rv)
                end
                while length(nv) >= n_samples && j < n
                    nvcalc = nv[1:n_samples]
                    if par
                        Folds.collect(write(fnm[i], string([j.val for j in run_teststatistic(t, nvcalc, m[i], s)], "\n")) for i in 1:length(m))
                    else
                        for i in 1:length(m)
                            write(fnm[i], string([j.val for j in run_teststatistic(t, nvcalc, m[i], s)], "\n"))
                        end
                    end
                    nv = length(nv) > n_samples ? nv[n_samples+1:end] : nv[1:0]
                    progress_every = max(1, fld(n_steps, 100))
                    j % progress_every == 0 && println(j)
                    j += 1
                end
            end
        end
    end
    close.(fnm)
end


"""
    read_teststatistic(t::AbstractTestcase, m::TM)
    read_teststatistic(t::AbstractTestcase, m::TM, s::AnySampler)

Functions to read test statistics from a file.
Read test statistics for a given test case, metric, and optionally a sampling algorithm from a file.

# Arguments
- `t::AbstractTestcase`: The test case for which the statistics are being read.
- `m::TM`: The metric, where `TM` is a subtype of `TestMetric`.
- `s::AnySampler` (optional): The sampling algorithm used to generate the statistics. If not givin the IID statistics are read.

# Returns
- `Array{Float64}`: A reshaped array containing the parsed test statistics.

# Notes
- Constructs the file name using `t.info` and `m.info`, and optionally `s.info` if a sampler is provided.
- Reads and parses the contents of the file to extract the statistics.

# Example
```julia
testcase = MyTestcase()
metric = MyMetric()
statistics = read_teststatistic(testcase, metric)  # Without sampler
println(statistics)

sampler = MySampler()
statistics_with_sampler = read_teststatistic(testcase, metric, sampler)  # With sampler
println(statistics_with_sampler)
```
"""
function read_teststatistic(t::AbstractTestcase, m::TM) where {TM <: TestMetric}
    filename = string(t.info, "-", m.info, ".txt")
    parse_teststatistic(string("./teststatistics/", filename))
end

function read_teststatistic(t::AbstractTestcase, m::TM, s::AnySampler) where {TM <: TestMetric}
    filename = string(t.info, "-", m.info, "-", s.info, ".txt")
    parse_teststatistic(string("./teststatistics_sampler/", filename))
end


"""
    parse_teststatistic(filename::String)

Parses test statistics from a specified file.

# Arguments
- `filename::String`: The name of the file containing test statistics.

# Returns
- `Array{Float64}`: A reshaped array containing the parsed test statistics.

# Notes
The function reads lines from the file, parses them as JSON, and reshapes the resulting data.

# Example
```julia
statistics = parse_teststatistic("./teststatistics/example.txt")
println(statistics)
```
"""
function parse_teststatistic(filename::String)
    f = open(string(filename), "r")
    mvals = JSON.parse.(readlines(f))
    close(f)
    reshape(vcat(mvals...), length(mvals[1]), length(mvals))
end


"""
    parseline(f::IOStream)

Parses a single line of JSON data from an open file stream.

# Arguments
- `f::IOStream`: The open file stream from which the line is read.

# Returns
- `Vector{Float64}`: A vector parsed from the JSON line.
- `[]`: An empty array if the line is empty.

# Example
```julia
file = open("./teststatistics/example.txt", "r")
line_data = parseline(file)
close(file)
println(line_data)
```
"""
function parseline(f::IOStream)
    iol = readline(f)
    if iol == ""
        []
    else
        return Vector{Float64}(JSON.parse(iol))
    end
end
