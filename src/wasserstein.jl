function _repeat_by_count(values, counts)
    reduce(vcat, (fill(value, count) for (value, count) in zip(values, counts)))
end

function _prepare_weights(values, weights, label)
    isnothing(weights) && return values, ones(length(values))
    length(weights) == length(values) || throw(DimensionMismatch(
        "$label weights must contain one entry per value",
    ))
    all(weight -> weight >= 0, weights) || throw(ArgumentError(
        "$label weights cannot be negative",
    ))

    positive = weights .> 0
    any(positive) || throw(ArgumentError("$label weights must have a positive sum"))
    values[positive], weights[positive]
end

"""
    wasserstein1d(a, b; p=1, wa=nothing, wb=nothing)

Compute the weighted one-dimensional Wasserstein distance of order `p`.
"""
function wasserstein1d(
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real};
    p::Real=1,
    wa=nothing,
    wb=nothing,
)
    isempty(a) && throw(ArgumentError("first input cannot be empty"))
    isempty(b) && throw(ArgumentError("second input cannot be empty"))
    p > 0 || throw(ArgumentError("p must be positive"))

    if length(a) == length(b) && isnothing(wa) && isnothing(wb)
        return mean(abs.(sort(b) .- sort(a)) .^ p)^(1 / p)
    end

    a, wa = _prepare_weights(a, wa, "first")
    b, wb = _prepare_weights(b, wb, "second")

    a_order = sortperm(a)
    b_order = sortperm(b)
    a = a[a_order]
    b = b[b_order]
    wa = wa[a_order]
    wb = wb[b_order]

    a_cumulative = cumsum((wa ./ sum(wa))[1:end-1])
    b_cumulative = cumsum((wb ./ sum(wb))[1:end-1])

    a_repeats = fit(
        Histogram,
        b_cumulative,
        vcat(-Inf, a_cumulative, Inf);
        closed=:left,
    ).weights .+ 1
    b_repeats = fit(
        Histogram,
        a_cumulative,
        vcat(-Inf, b_cumulative, Inf);
        closed=:left,
    ).weights .+ 1

    repeated_a = _repeat_by_count(a, a_repeats)
    repeated_b = _repeat_by_count(b, b_repeats)
    cumulative = sort(vcat(a_cumulative, b_cumulative))
    interval_widths = vcat(cumulative, 1) .- vcat(0, cumulative)

    sum(interval_widths .* abs.(repeated_b .- repeated_a) .^ p)^(1 / p)
end

"""Generate a uniformly oriented unit vector in `n` dimensions."""
function get_random_projection(n::Int)
    n > 0 || throw(ArgumentError("projection dimension must be positive"))
    projection = randn(n)
    projection / norm(projection)
end

function project_sample(
    sample::AbstractVector{<:Real},
    axis::AbstractVector{<:Real},
)
    length(sample) == length(axis) || throw(DimensionMismatch(
        "sample and projection axis must have the same dimension",
    ))
    dot(sample, axis)
end

project_samples(samples, axis::AbstractVector{<:Real}) =
    project_sample.(samples, Ref(axis))

function _projected_wasserstein(
    first_values::AbstractMatrix,
    second_values::AbstractMatrix,
    axis::AbstractVector,
    first_weights,
    second_weights,
    weighted::Bool,
    p::Real,
)
    first_projection = vec(transpose(axis) * first_values)
    second_projection = vec(transpose(axis) * second_values)

    if weighted
        return wasserstein1d(
            first_projection,
            second_projection;
            p=p,
            wa=first_weights,
            wb=second_weights,
        )
    end
    wasserstein1d(first_projection, second_projection; p=p)
end

"""
    get_sliced_wasserstein_distance(first, second;
                                    L=1000, p=1, N=0, parallel=true)

Average one-dimensional Wasserstein distances over `L` random projections.
"""
function get_sliced_wasserstein_distance(
    first::DensitySampleVector,
    second::DensitySampleVector;
    L::Int=1_000,
    p::Real=1,
    N::Int=0,
    parallel::Bool=true,
)
    L > 0 || throw(ArgumentError("L must be positive"))
    p > 0 || throw(ArgumentError("p must be positive"))
    first, second = prepare_twosample_dsv(first, second; N=N)

    first_values = _sample_value_matrix(first)
    second_values = _sample_value_matrix(second)
    size(first_values, 1) == size(second_values, 1) || throw(DimensionMismatch(
        "sliced Wasserstein inputs must have the same dimension",
    ))

    dimension = size(first_values, 1)
    weighted = is_weighted(first) || is_weighted(second)
    distances = Vector{Float64}(undef, L)

    if parallel && Threads.nthreads() > 1
        Threads.@threads for index in eachindex(distances)
            axis = get_random_projection(dimension)
            distances[index] = _projected_wasserstein(
                first_values,
                second_values,
                axis,
                first.weight,
                second.weight,
                weighted,
                p,
            )
        end
    else
        for index in eachindex(distances)
            axis = get_random_projection(dimension)
            distances[index] = _projected_wasserstein(
                first_values,
                second_values,
                axis,
                first.weight,
                second.weight,
                weighted,
                p,
            )
        end
    end

    norm(distances, p) / length(distances)^(1 / p)
end

# Kept as a small public helper for code that previously called it directly.
function i_get_sliced_wasserstein_distance(
    dimension,
    first::DensitySampleVector,
    second::DensitySampleVector,
)
    axis = get_random_projection(dimension)
    _projected_wasserstein(
        _sample_value_matrix(first),
        _sample_value_matrix(second),
        axis,
        first.weight,
        second.weight,
        is_weighted(first) || is_weighted(second),
        1,
    )
end

"""Estimate sliced Wasserstein distance directly from two distributions."""
function get_sliced_wasserstein_distance(
    first_distribution,
    second_distribution;
    d::Int=50,
    L::Int=1_000,
    p::Real=1,
    N::Int=100_000,
)
    d > 0 || throw(ArgumentError("d must be positive"))
    N > 0 || throw(ArgumentError("N must be positive"))
    L > 0 || throw(ArgumentError("L must be positive"))

    first_samples = [rand(first_distribution, d) for _ in 1:N]
    second_samples = [rand(second_distribution, d) for _ in 1:N]
    distances = Vector{Float64}(undef, L)

    for index in eachindex(distances)
        axis = get_random_projection(d)
        first_projection = project_samples(first_samples, axis)
        second_projection = project_samples(second_samples, axis)
        distances[index] = wasserstein1d(first_projection, second_projection; p=p)
    end

    norm(distances, p) / length(distances)^(1 / p)
end

export wasserstein1d, get_random_projection, project_sample, project_samples
export get_sliced_wasserstein_distance
