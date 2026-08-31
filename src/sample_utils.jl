function _validate_dsv_inputs(samples, logdensities, weights)
    sample_count = size(samples, 2)
    length(logdensities) == sample_count || throw(DimensionMismatch(
        "received $sample_count samples but $(length(logdensities)) log densities",
    ))
    length(weights) == sample_count || throw(DimensionMismatch(
        "received $sample_count samples but $(length(weights)) weights",
    ))
    nothing
end

function _sample_coordinates(value)
    if hasproperty(value, :x)
        coordinates = getproperty(value, :x)
        return coordinates isa Real ? [coordinates] : coordinates
    end
    BAT.unshaped(value)
end

function _sample_value_matrix(samples::DensitySampleVector)
    Matrix{Float64}(hcat(_sample_coordinates.(samples.v)...))
end

"""
    make_dsv(samples; logdensities=ones(size(samples, 2)), weights=...)
    make_dsv(samples, logdensities; weights=...)

Create a BAT `DensitySampleVector` from a matrix whose columns are samples.
Log densities are optional when downstream metrics do not require them.
"""
function make_dsv(
    samples::AbstractMatrix{<:Real};
    logdensities::AbstractVector{<:Real}=ones(size(samples, 2)),
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    make_dsv(samples, logdensities; weights=weights)
end

function make_dsv(
    samples::AbstractMatrix{<:Real},
    logdensities::AbstractVector{<:Real};
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    _validate_dsv_inputs(samples, logdensities, weights)
    values = [collect(samples[:, index]) for index in axes(samples, 2)]
    DensitySampleVector(values, logdensities; weight=weights)
end

function make_dsv(
    samples::AbstractMatrix{<:Real},
    logdensities::AbstractVector{<:Real},
    infos;
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    _validate_dsv_inputs(samples, logdensities, weights)
    length(infos) == size(samples, 2) || throw(DimensionMismatch(
        "info must contain one entry per sample",
    ))
    values = [collect(samples[:, index]) for index in axes(samples, 2)]
    DensitySampleVector(values, logdensities; weight=weights, info=infos)
end

function make_dsv(
    samples::AbstractVector{<:Real},
    logdensities::AbstractVector{<:Real};
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    make_dsv(reshape(samples, 1, :), logdensities; weights=weights)
end

function make_dsv(
    samples::AbstractVector{<:AbstractVector{<:Real}},
    logdensities::AbstractVector{<:Real};
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    isempty(samples) && throw(ArgumentError("samples cannot be empty"))
    dimensions = unique(length.(samples))
    length(dimensions) == 1 || throw(DimensionMismatch(
        "all sample vectors must have the same dimension",
    ))
    make_dsv(hcat(samples...), logdensities; weights=weights)
end

function make_dsv(
    samples::AbstractVector{<:AbstractVector{<:Real}};
    logdensities::AbstractVector{<:Real}=ones(length(samples)),
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    make_dsv(samples, logdensities; weights=weights)
end

function make_dsv(
    samples::AbstractVector{<:Real};
    logdensities::AbstractVector{<:Real}=ones(length(samples)),
    weights::AbstractVector{<:Real}=ones(length(logdensities)),
)
    make_dsv(samples, logdensities; weights=weights)
end

export make_dsv


"""Return the Kish effective sample size of a density sample vector."""
function get_effective_sample_size(dsv::DensitySampleVector)
    BAT.bat_eff_sample_size(dsv, BAT.KishESS()).result
end

get_effective_sample_size(dsv::DensitySampleVector, ::Int) = get_effective_sample_size(dsv)

function _minimum_effective_sample_size(result)
    result isa Real && return Float64(result)
    if result isa NamedTuple
        values_by_dimension = Iterators.flatten(
            value isa Real ? (value,) : value for value in values(result)
        )
        return Float64(minimum(values_by_dimension))
    end
    Float64(minimum(result))
end

"""
    get_effective_sample_size(samples, sampler)

Return a conservative scalar effective sample size for `samples`. IID
samplers use the weight-based Kish ESS. Other sampling algorithms use BAT's
autocorrelation estimate and return the smallest ESS across dimensions.
"""
function get_effective_sample_size(
    dsv::DensitySampleVector,
    ::IIDSamplingAlgorithm,
)
    get_effective_sample_size(dsv)
end

function get_effective_sample_size(
    dsv::DensitySampleVector,
    ::SamplingAlgorithm,
)
    result = BAT.bat_eff_sample_size(
        dsv,
        BAT.EffSampleSizeFromAC(),
    ).result
    _minimum_effective_sample_size(result)
end

"""Whether at least one sample has a weight other than one."""
is_weighted(dsv::DensitySampleVector) = any(weight -> !isone(weight), dsv.weight)

"""Replace repeated unweighted samples with one sample and an integer weight."""
function condense_dsv(dsv::DensitySampleVector)
    is_weighted(dsv) && throw(ArgumentError("sample vector is already weighted"))

    values = BAT.unshaped.(dsv.v)
    indices, weights = BAT.repetition_to_weights(values)
    DensitySampleVector(
        values[indices],
        dsv.logd[indices];
        weight=weights,
        info=dsv.info[indices],
        aux=dsv.aux[indices],
    )
end

function resample_dsv_to_ess(dsv::DensitySampleVector)
    sample_count = floor(Int, get_effective_sample_size(dsv))
    resample_dsv(dsv, sample_count)
end

function resample_dsv_to_ess(
    dsv::DensitySampleVector,
    sampler::SamplingAlgorithm,
)
    sample_count = floor(Int, get_effective_sample_size(dsv, sampler))
    resample_dsv(dsv, sample_count)
end

function resample_dsv(dsv::DensitySampleVector, n::Int)
    n >= 0 || throw(ArgumentError("resample size cannot be negative"))
    bat_sample(dsv, RandResampling(nsamples=n)).result
end

export get_effective_sample_size, is_weighted, condense_dsv
export resample_dsv_to_ess, resample_dsv


"""
    prepare_twosample_dsv(first, second; N=0)

Return two sample vectors with compatible sizes. When their lengths differ,
both are resampled to the smaller effective sample size. A positive `N` asks
for an explicit common size and is capped at the available effective size.
"""
function prepare_twosample_dsv(
    first::DensitySampleVector,
    second::DensitySampleVector;
    N::Int=0,
)
    N >= 0 || throw(ArgumentError("N cannot be negative"))
    first_ess = get_effective_sample_size(first)
    second_ess = get_effective_sample_size(second)
    max_common_size = floor(Int, min(first_ess, second_ess))

    sample_count = N
    if length(first) != length(second)
        sample_count = iszero(N) ? max_common_size : min(N, max_common_size)
        @warn "Two-sample metrics require equal lengths; resampling both inputs" sample_count
    elseif N > max_common_size
        sample_count = max_common_size
        @warn "Requested two-sample size exceeds the effective sample size; reducing N" requested=N sample_count
    end

    iszero(sample_count) && return first, second
    sample_count > 0 || throw(ArgumentError("effective sample size must be positive"))

    resample_dsv(first, sample_count), resample_dsv(second, sample_count)
end

export prepare_twosample_dsv
