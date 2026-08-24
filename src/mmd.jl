"""Base type for kernels used by maximum mean discrepancy."""
abstract type AbstractKernel end

const MetricOrFunction = Union{PreMetric,Function}

pairwisel2(x::AbstractMatrix, y::AbstractMatrix) =
    pairwise(SqEuclidean(), x, y; dims=2)
pairwisel2(x::AbstractMatrix) = pairwisel2(x, x)

function kernelsum(
    kernel::AbstractKernel,
    x::AbstractMatrix,
    y::AbstractMatrix,
    distance::MetricOrFunction,
)
    sum(kernel(distance(x, y))) / (size(x, 2) * size(y, 2))
end

function kernelsum(
    kernel::AbstractKernel,
    x::AbstractMatrix{T},
    distance::MetricOrFunction,
) where {T}
    sample_count = size(x, 2)
    sample_count > 1 || throw(ArgumentError("MMD requires at least two samples"))

    diagonal = sample_count * kernel(zero(T))
    (sum(kernel(distance(x, x))) - diagonal) / (sample_count^2 - sample_count)
end

kernelsum(::AbstractKernel, x::AbstractVector, ::MetricOrFunction) = zero(eltype(x))

"""Gaussian kernel `exp(-γ * squared_distance)`."""
struct GaussianKernel{T<:Real} <: AbstractKernel
    γ::T

    function GaussianKernel(γ::T) where {T<:Real}
        γ >= 0 || throw(ArgumentError("Gaussian-kernel gamma cannot be negative"))
        new{T}(γ)
    end
end

(kernel::GaussianKernel)(value::Number) = exp(-kernel.γ * value)
(kernel::GaussianKernel)(values::AbstractArray) = exp.(-kernel.γ .* values)

struct MMD{K<:AbstractKernel,D<:MetricOrFunction} <: PreMetric
    kernel::K
    distance::D
end

function (metric::MMD)(x::AbstractArray, y::AbstractArray)
    within_x = kernelsum(metric.kernel, x, metric.distance)
    within_y = kernelsum(metric.kernel, y, metric.distance)
    between = kernelsum(metric.kernel, x, y, metric.distance)
    within_x + within_y - 2between
end

function mmd(
    kernel::AbstractKernel,
    x::AbstractArray,
    y::AbstractArray,
    distance=pairwisel2,
)
    MMD(kernel, distance)(x, y)
end

function mmd(
    kernel::AbstractKernel,
    x::AbstractMatrix,
    y::AbstractMatrix,
    n::Int,
    distance=pairwisel2,
)
    n > 1 || throw(ArgumentError("MMD subsample size must be at least two"))
    n <= min(size(x, 2), size(y, 2)) || throw(DimensionMismatch(
        "MMD subsample size exceeds the available columns",
    ))

    x_columns = randperm(size(x, 2))[1:n]
    y_columns = randperm(size(y, 2))[1:n]
    mmd(kernel, x[:, x_columns], y[:, y_columns], distance)
end

"""Calculate MMD between two BAT density sample vectors."""
function get_mmd(
    first::DensitySampleVector,
    second::DensitySampleVector;
    g::Real=0,
    N::Int=0,
)
    first, second = prepare_twosample_dsv(first, second; N=N)
    get_mmd(_sample_value_matrix(first), _sample_value_matrix(second); g=g)
end

function get_mmd(x::AbstractMatrix, y::AbstractMatrix; g::Real=0)
    size(x, 1) == size(y, 1) || throw(DimensionMismatch(
        "MMD inputs must have the same number of dimensions",
    ))
    gamma = iszero(g) ? compute_bandwidth(x, y) : g
    mmd(GaussianKernel(gamma), x, y)
end

"""Median pairwise squared distance used as the Gaussian-kernel scale."""
function compute_bandwidth(x::AbstractMatrix, y::AbstractMatrix)
    size(x, 1) == size(y, 1) || throw(DimensionMismatch(
        "bandwidth inputs must have the same number of dimensions",
    ))
    combined = hcat(x, y)
    size(combined, 2) > 1 || throw(ArgumentError(
        "bandwidth estimation requires at least two samples",
    ))

    distances = pairwise(SqEuclidean(), combined; dims=2)
    upper_triangle = distances[triu(trues(size(distances)), 1)]
    median(upper_triangle)
end

function compute_bandwidth(
    first::DensitySampleVector,
    second::DensitySampleVector,
)
    first, second = prepare_twosample_dsv(first, second)
    compute_bandwidth(_sample_value_matrix(first), _sample_value_matrix(second))
end

export get_mmd, compute_bandwidth
