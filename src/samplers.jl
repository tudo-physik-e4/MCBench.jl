"""Base type for every sampler accepted by MCBench."""
abstract type AnySampler end

"""Base type for sampling algorithms that generate samples from a target."""
abstract type SamplingAlgorithm <: AnySampler end

"""Marker type for independent and identically distributed samplers."""
abstract type IIDSamplingAlgorithm <: SamplingAlgorithm end

"""Base type for samplers that read or resample stored values."""
abstract type AbstractFileBasedSampler <: AnySampler end

export AnySampler, SamplingAlgorithm, IIDSamplingAlgorithm, AbstractFileBasedSampler


"""
    IIDSampler(n_steps=100_000, info="IID")

Configuration object used when MCBench should draw IID target samples.
"""
struct IIDSampler <: IIDSamplingAlgorithm
    n_steps::Int
    info::String

    function IIDSampler(n_steps::Int, info::String)
        n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
        new(n_steps, info)
    end
end

IIDSampler() = IIDSampler(100_000, "IID")

export IIDSampler


"""
    FileBasedSampler(paths; info="FileBasedSampler")
    FileBasedSampler(path; info="FileBasedSampler")

Read one sample per line from one or more plain-text files. Directory inputs
are expanded to their contained files in lexical order.
"""
mutable struct FileBasedSampler <: AbstractFileBasedSampler
    files::Vector{String}
    current_file_index::Int
    current_position::Int
    current_file_handle::IOStream
    info::String
end

function FileBasedSampler(
    file_paths::AbstractVector{<:AbstractString};
    info::String="FileBasedSampler",
)
    isempty(file_paths) && throw(ArgumentError("file paths cannot be empty"))
    all(isfile, file_paths) || throw(ArgumentError("every sampler path must be a file"))

    files = String.(abspath.(file_paths))
    handle = open(first(files), "r")
    FileBasedSampler(files, 1, 0, handle, info)
end

function FileBasedSampler(path::AbstractString; info::String="FileBasedSampler")
    files = if isfile(path)
        [path]
    elseif isdir(path)
        sort(filter(isfile, readdir(path; join=true)))
    else
        throw(ArgumentError("path does not exist or is not a file or directory: $path"))
    end

    isempty(files) && throw(ArgumentError("no files found at: $path"))
    FileBasedSampler(files; info=info)
end

function _advance_file!(sampler::FileBasedSampler)
    close(sampler.current_file_handle)
    sampler.current_file_index += 1

    if sampler.current_file_index > length(sampler.files)
        error("no more sample files to read")
    end

    sampler.current_file_handle = open(sampler.files[sampler.current_file_index], "r")
    sampler.current_position = 0
    sampler
end

function read_sample!(sampler::FileBasedSampler)
    # Empty files are skipped so a directory may safely contain placeholders.
    while eof(sampler.current_file_handle)
        _advance_file!(sampler)
    end

    line = readline(sampler.current_file_handle)
    sampler.current_position += 1
    line
end

function reset_sampler!(sampler::FileBasedSampler)
    close(sampler.current_file_handle)
    sampler.current_file_index = 1
    sampler.current_position = 0
    sampler.current_file_handle = open(first(sampler.files), "r")
    sampler
end

function close_sampler!(sampler::FileBasedSampler)
    isopen(sampler.current_file_handle) && close(sampler.current_file_handle)
    nothing
end

export FileBasedSampler


"""
    CsvBasedSampler(paths; info="CsvBasedSampler")
    CsvBasedSampler(path; info="CsvBasedSampler")

Read numeric CSV rows while retaining the header and a selectable column mask.
Each input file must use the same header. By default all columns are read in
their file order. Use [`set_mask`](@ref) to select and reorder parameter
columns before sampling.
"""
mutable struct CsvBasedSampler <: AbstractFileBasedSampler
    fbs::FileBasedSampler
    header::Vector{String}
    mask::Vector{Int}
    info::String
end

function _csv_sampler(file_sampler::FileBasedSampler, info::String)
    header = String.(split(read_sample!(file_sampler), ","))
    CsvBasedSampler(file_sampler, header, collect(eachindex(header)), info)
end

function CsvBasedSampler(
    file_paths::AbstractVector{<:AbstractString};
    info::String="CsvBasedSampler",
)
    _csv_sampler(FileBasedSampler(file_paths; info=info), info)
end

function CsvBasedSampler(path::AbstractString; info::String="CsvBasedSampler")
    _csv_sampler(FileBasedSampler(path; info=info), info)
end

function read_sample!(sampler::CsvBasedSampler)
    while true
        previous_file = sampler.fbs.current_file_index
        line = read_sample!(sampler.fbs)
        sampler.fbs.current_file_index == previous_file && return line

        # The first row after a file transition is its header. Validate and
        # continue so header-only files are handled correctly as well.
        next_header = String.(split(line, ","))
        previous_path = sampler.fbs.files[previous_file]
        current_path = sampler.fbs.files[sampler.fbs.current_file_index]
        next_header == sampler.header || throw(ArgumentError(
            "CSV headers differ between $previous_path and $current_path",
        ))
    end
end

"""
    set_mask(sampler::CsvBasedSampler, columns)

Select CSV columns by header name and arrange them in the order given by
`columns`. The updated sampler is returned, so the call may be chained.
"""
function set_mask(sampler::CsvBasedSampler, columns::AbstractVector{<:AbstractString})
    header_indices = Dict(name => index for (index, name) in enumerate(sampler.header))
    missing_columns = filter(column -> !haskey(header_indices, column), columns)
    isempty(missing_columns) || throw(ArgumentError(
        "columns not found in CSV header: $(join(missing_columns, ", "))",
    ))

    sampler.mask = [header_indices[column] for column in columns]
    sampler
end

function reset_sampler!(sampler::CsvBasedSampler)
    reset_sampler!(sampler.fbs)
    header = String.(split(read_sample!(sampler.fbs), ","))
    header == sampler.header || throw(ArgumentError("CSV header changed after reset"))
    sampler
end

close_sampler!(sampler::CsvBasedSampler) = close_sampler!(sampler.fbs)

export CsvBasedSampler


"""
    DsvSampler(dsvs; info="DsvSampler")

Store one or more BAT `DensitySampleVector` objects for deterministic reuse and
resampling in benchmark workflows. `info` is used in output filenames, plot
labels, and summary headings. MCBench resamples a stored vector when an
unweighted batch with a requested size is needed.
"""
mutable struct DsvSampler{D<:DensitySampleVector} <: AbstractFileBasedSampler
    dsvs::Vector{D}
    current_dsv_index::Int
    current_position::Int
    weighted::Bool
    total_samples::Int
    neff::Vector{Float64}
    info::String
end

function DsvSampler(
    dsvs::Vector{D};
    info::String="DsvSampler",
) where {D<:DensitySampleVector}
    isempty(dsvs) && throw(ArgumentError("density sample vectors cannot be empty"))
    any(isempty, dsvs) && throw(ArgumentError("density sample vectors cannot contain empty samples"))

    weighted = any(is_weighted, dsvs)
    total_samples = sum(length, dsvs)
    effective_sizes = Float64[get_effective_sample_size(dsv) for dsv in dsvs]
    DsvSampler(dsvs, 1, 1, weighted, total_samples, effective_sizes, info)
end

function read_sample!(sampler::DsvSampler)
    if sampler.current_position > length(sampler.dsvs[sampler.current_dsv_index])
        sampler.current_dsv_index += 1
        sampler.current_position = 1
    end

    if sampler.current_dsv_index > length(sampler.dsvs)
        error("no more density samples to read")
    end

    value = sampler.dsvs[sampler.current_dsv_index][sampler.current_position]
    sampler.current_position += 1
    value
end

function reset_sampler!(sampler::DsvSampler)
    sampler.current_dsv_index = 1
    sampler.current_position = 1
    sampler
end

function unweight!(sampler::DsvSampler)
    sampler.weighted || return sampler

    for index in eachindex(sampler.dsvs)
        sampler.dsvs[index] = resample_dsv_to_ess(sampler.dsvs[index])
    end
    sampler.weighted = false
    sampler.total_samples = sum(length, sampler.dsvs)
    sampler.neff = Float64[get_effective_sample_size(dsv) for dsv in sampler.dsvs]
    reset_sampler!(sampler)
end

export DsvSampler
