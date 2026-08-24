"""
    abstract type Target

Base type for custom target distributions used by MCBench. A concrete target
must implement `rand(target, n)`, `length(target)`, and `logpdf(target, values)`.
"""
abstract type Target end

"""Base type for all MCBench testcases."""
abstract type AbstractTestcase end

const _EMPTY_REFERENCE_VALUES = NamedTuple()

function _validate_reference_values(values)
    values isa NamedTuple || throw(ArgumentError("reference_values must be a NamedTuple"))

    for (name, value) in pairs(values)
        valid = value isa Real || value isa AbstractVector{<:Real} || value isa Function
        valid || throw(ArgumentError(
            "reference value :$name must be numeric, a numeric vector, or a function",
        ))
        value isa Function && continue
        all(isfinite, value isa Real ? (value,) : value) || throw(ArgumentError(
            "reference value :$name must contain only finite numbers",
        ))
    end

    values
end

"""
    Testcases(f, bounds, dim, info; reference_values=(;))
    Testcases(f, dim, info; reference_values=(;))

A sampleable target distribution together with its bounds, dimensionality, and
display name. Optional `reference_values` attach known population values to the
testcase. Keys are metric type names, for example:

```julia
Testcases(
    MvNormal(zeros(2), I(2)),
    bounds,
    2,
    "Standard-Normal";
    reference_values=(
        marginal_mean=zeros(2),
        marginal_variance=ones(2),
    ),
)
```

Scalars may be used when all dimensions share the same reference value. For a
configurable metric, an entry may instead be a function that receives the
metric and returns its matching scalar or vector.
"""
struct Testcases{
    D<:Union{Distribution,Target},
    B<:NamedTupleDist,
    A,
    N<:Int,
    R<:NamedTuple,
} <: AbstractTestcase
    f::D
    bounds::B
    dim::N
    info::A
    reference_values::R
end

function Testcases(
    f::D,
    bounds::B,
    dim::N,
    info::A;
    reference_values=_EMPTY_REFERENCE_VALUES,
) where {D<:Union{Distribution,Target},B<:NamedTupleDist,A,N<:Int}
    dim > 0 || throw(ArgumentError("testcase dimension must be positive"))
    references = _validate_reference_values(reference_values)
    Testcases(f, bounds, dim, info, references)
end

function Testcases(
    f::D,
    dim::N,
    info::A;
    reference_values=_EMPTY_REFERENCE_VALUES,
) where {D<:Union{Distribution,Target},A,N<:Int}
    bounds = NamedTupleDist(x=fill(-10..10, dim))
    Testcases(f, bounds, dim, info; reference_values=reference_values)
end

export Testcases

@inline DensityInterface.DensityKind(::Testcases) = IsDensity()

function _sample_logdensities(distribution, values)
    if distribution isa UnivariateDistribution
        return logpdf.(Ref(distribution), vec(values))
    end
    logpdf(distribution, values)
end

"""
    sample(testcase::Testcases; n_steps=100_000)
    sample(testcase::Testcases, n::Int)

Draw IID samples from a testcase and return a `DensitySampleVector` containing
both values and their target log densities.
"""
function sample(testcase::Testcases; n_steps::Int=100_000)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    values = rand(testcase.f, n_steps)
    logdensities = _sample_logdensities(testcase.f, values)
    make_dsv(values, logdensities)
end

sample(testcase::Testcases, n::Int) = sample(testcase; n_steps=n)

function sample(
    testcase::Testcases,
    ::IIDSamplingAlgorithm;
    n_steps::Int=100_000,
)
    sample(testcase; n_steps=n_steps)
end

function sample(
    testcase::Testcases,
    sampler::IIDSampler;
    n_steps::Int=sampler.n_steps,
)
    sample(testcase; n_steps=n_steps)
end

# File-backed samplers own the sample generation. The testcase is passed only
# so samplers can evaluate target log densities when possible.
function sample(
    testcase::AbstractTestcase,
    sampler::AbstractFileBasedSampler;
    n_steps::Int=10_000,
)
    sample(sampler; t=testcase, n_steps=n_steps)
end

function _file_sample_logdensities(testcase, values::AbstractMatrix)
    if testcase isa Testcases
        return _sample_logdensities(testcase.f, values)
    end
    ones(size(values, 2))
end

function sample(
    sampler::FileBasedSampler;
    t=nothing,
    n_steps::Int=10_000,
)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    rows = [parse.(Float64, split(read_sample!(sampler), ",")) for _ in 1:n_steps]
    values = hcat(rows...)
    make_dsv(values, _file_sample_logdensities(t, values))
end

function sample(sampler::CsvBasedSampler, n::Int)
    n > 0 || throw(ArgumentError("sample count must be positive"))
    rows = Vector{Float64}[]
    sizehint!(rows, n)

    for _ in 1:n
        fields = split(read_sample!(sampler), ",")
        push!(rows, parse.(Float64, fields[sampler.mask]))
    end

    make_dsv(rows)
end

function sample(
    sampler::CsvBasedSampler;
    t=nothing,
    n_steps::Int=10_000,
)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    rows = [
        parse.(Float64, split(read_sample!(sampler), ",")[sampler.mask])
        for _ in 1:n_steps
    ]
    values = hcat(rows...)
    make_dsv(values, _file_sample_logdensities(t, values))
end

function sample(
    sampler::DsvSampler;
    t=nothing,
    n_steps::Int=10_000,
)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    available = floor(Int, sampler.neff[sampler.current_dsv_index])
    sample_count = min(n_steps, available)

    if sample_count < n_steps
        message = "Requested more samples than the available effective sample size; reducing the sample count"
        @warn message requested=n_steps available=available
    end

    resample_dsv(sampler.dsvs[sampler.current_dsv_index], sample_count)
end

"""
    DsvTestcase(sampler, dim, info; reference_values=(;))
    DsvTestcase(sampler; n=0, info="DsvTestcase", reference_values=(;))

A testcase backed by precomputed `DensitySampleVector` objects. When `n` is
zero, the dimensionality is inferred from the first stored sample.
"""
struct DsvTestcase{
    DS<:DsvSampler,
    A,
    N<:Int,
    R<:NamedTuple,
} <: AbstractTestcase
    sampler::DS
    dim::N
    info::A
    reference_values::R
end

function DsvTestcase(
    sampler::DS,
    dim::N,
    info::A;
    reference_values=_EMPTY_REFERENCE_VALUES,
) where {DS<:DsvSampler,A,N<:Int}
    dim > 0 || throw(ArgumentError("testcase dimension must be positive"))
    references = _validate_reference_values(reference_values)
    DsvTestcase(sampler, dim, info, references)
end

function DsvTestcase(
    sampler::DS;
    n::Int=0,
    info="DsvTestcase",
    reference_values=_EMPTY_REFERENCE_VALUES,
) where {DS<:DsvSampler}
    inferred_dim = length(BAT.unshaped(sampler.dsvs[1].v[1]))
    dim = iszero(n) ? inferred_dim : n
    DsvTestcase(sampler, dim, info; reference_values=reference_values)
end

export DsvTestcase

sample(testcase::DsvTestcase, n::Int) = sample(testcase.sampler; t=testcase, n_steps=n)

function sample(testcase::DsvTestcase; n_steps::Int=100_000)
    sample(testcase.sampler; t=testcase, n_steps=n_steps)
end

# General algorithms cannot generate from a sample-backed target, so its own
# DSV sampler remains the source of IID reference samples.
function sample(
    testcase::DsvTestcase,
    ::SamplingAlgorithm;
    n_steps::Int=100_000,
)
    sample(testcase; n_steps=n_steps)
end

export sample
