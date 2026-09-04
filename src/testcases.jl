"""
    abstract type Target

Base type for custom target distributions used by MCBench. A concrete target
must implement `rand(target, n)`, `length(target)`, and `logpdf(target, values)`.
"""
abstract type Target end

"""Base type for all MCBench testcases."""
abstract type AbstractTestcase end

const _EMPTY_REFERENCE_VALUES = NamedTuple()
const _EMPTY_REFERENCE_DISTRIBUTIONS = NamedTuple()

"""
    AnalyticReferenceDistribution(observable, distribution;
                                  info="Analytic observable",
                                  goodness_of_fit=one_sample_ks_test,
                                  test_name=nothing)

Describe an observation-level scalar `observable` and its analytically known
univariate `distribution` under the testcase target. The observable receives
one sample as a coordinate vector and must return one finite real number.

`goodness_of_fit(values, distribution)` compares all transformed values with
the analytic distribution and must return a named tuple containing finite
`statistic` and `pvalue` fields. The default is an asymptotic one-sample KS
test for continuous distributions. When an ESS-adjusted result is requested,
a custom function must additionally support
`goodness_of_fit(values, distribution, effective_sample_size)`. This keeps the
same interface usable for discrete distributions, such as an exactly known
Ising energy law, while leaving their appropriate correction to the custom
test.
"""
struct AnalyticReferenceDistribution{O,D<:UnivariateDistribution,G}
    observable::O
    distribution::D
    info::String
    goodness_of_fit::G
    test_name::String
end

function AnalyticReferenceDistribution(
    observable,
    distribution::UnivariateDistribution;
    info="Analytic observable",
    goodness_of_fit=one_sample_ks_test,
    test_name=nothing,
)
    resolved_test_name = if isnothing(test_name)
        goodness_of_fit === one_sample_ks_test ?
            "One-sample KS" : "Goodness-of-fit test"
    else
        string(test_name)
    end
    AnalyticReferenceDistribution(
        observable,
        distribution,
        string(info),
        goodness_of_fit,
        resolved_test_name,
    )
end

function _validate_reference_distributions(references)
    references isa NamedTuple || throw(ArgumentError(
        "reference_distributions must be a NamedTuple",
    ))
    for (name, reference) in pairs(references)
        reference isa AnalyticReferenceDistribution || throw(ArgumentError(
            "reference distribution :$name must be an AnalyticReferenceDistribution",
        ))
    end
    references
end

export AnalyticReferenceDistribution

_is_reference_element(value) = value isa Real || ismissing(value)

function _validate_reference_values(values)
    values isa NamedTuple || throw(ArgumentError("reference_values must be a NamedTuple"))

    for (name, value) in pairs(values)
        valid = _is_reference_element(value) || value isa Function ||
            (value isa AbstractVector && all(_is_reference_element, value))
        valid || throw(ArgumentError(
            "reference value :$name must be numeric, missing, a vector of numeric or missing values, or a function",
        ))
        value isa Function && continue
        elements = value isa AbstractVector ? value : (value,)
        all(element -> ismissing(element) || isfinite(element), elements) || throw(ArgumentError(
            "reference value :$name must contain only finite numbers or missing",
        ))
    end

    values
end

"""
    Testcases(f, bounds, dim, info;
              reference_values=(;), reference_distributions=(;))
    Testcases(f, dim, info;
              reference_values=(;), reference_distributions=(;))

A sampleable target distribution together with its bounds, dimensionality, and
display name. Optional `reference_values` attach known scalar population
values to the testcase. Keys are metric type names, for example:

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

Scalars may be used when all dimensions share the same reference value. Use
`missing` inside a vector when references are known for only some metric
outputs. For a configurable metric, an entry may instead be a function that
receives the metric and returns its matching scalar or vector.

The separate `reference_distributions` keyword stores analytically known
distributions of observation-level scalar transforms. See
[`AnalyticReferenceDistribution`](@ref). These are intentionally not scalar
metric reference values and do not require IID reference samples.
"""
struct Testcases{
    D<:Union{Distribution,Target},
    B<:NamedTupleDist,
    A,
    N<:Int,
    R<:NamedTuple,
    RD<:NamedTuple,
} <: AbstractTestcase
    f::D
    bounds::B
    dim::N
    info::A
    reference_values::R
    reference_distributions::RD
end

function Testcases(
    f::D,
    bounds::B,
    dim::N,
    info::A;
    reference_values=_EMPTY_REFERENCE_VALUES,
    reference_distributions=_EMPTY_REFERENCE_DISTRIBUTIONS,
) where {D<:Union{Distribution,Target},B<:NamedTupleDist,A,N<:Int}
    dim > 0 || throw(ArgumentError("testcase dimension must be positive"))
    references = _validate_reference_values(reference_values)
    distributions = _validate_reference_distributions(reference_distributions)
    Testcases(f, bounds, dim, info, references, distributions)
end

function Testcases(
    f::D,
    dim::N,
    info::A;
    reference_values=_EMPTY_REFERENCE_VALUES,
    reference_distributions=_EMPTY_REFERENCE_DISTRIBUTIONS,
) where {D<:Union{Distribution,Target},A,N<:Int}
    bounds = NamedTupleDist(x=fill(-10..10, dim))
    Testcases(
        f,
        bounds,
        dim,
        info;
        reference_values=reference_values,
        reference_distributions=reference_distributions,
    )
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
    DsvTestcase(sampler, dim, info;
                reference_values=(;), reference_distributions=(;))
    DsvTestcase(sampler; n=0, info="DsvTestcase",
                reference_values=(;), reference_distributions=(;))

A testcase backed by precomputed `DensitySampleVector` objects. When `n` is
zero, the dimensionality is inferred from the first stored sample.
"""
struct DsvTestcase{
    DS<:DsvSampler,
    A,
    N<:Int,
    R<:NamedTuple,
    RD<:NamedTuple,
} <: AbstractTestcase
    sampler::DS
    dim::N
    info::A
    reference_values::R
    reference_distributions::RD
end

function DsvTestcase(
    sampler::DS,
    dim::N,
    info::A;
    reference_values=_EMPTY_REFERENCE_VALUES,
    reference_distributions=_EMPTY_REFERENCE_DISTRIBUTIONS,
) where {DS<:DsvSampler,A,N<:Int}
    dim > 0 || throw(ArgumentError("testcase dimension must be positive"))
    references = _validate_reference_values(reference_values)
    distributions = _validate_reference_distributions(reference_distributions)
    DsvTestcase(sampler, dim, info, references, distributions)
end

function DsvTestcase(
    sampler::DS;
    n::Int=0,
    info="DsvTestcase",
    reference_values=_EMPTY_REFERENCE_VALUES,
    reference_distributions=_EMPTY_REFERENCE_DISTRIBUTIONS,
) where {DS<:DsvSampler}
    inferred_dim = length(BAT.unshaped(sampler.dsvs[1].v[1]))
    dim = iszero(n) ? inferred_dim : n
    DsvTestcase(
        sampler,
        dim,
        info;
        reference_values=reference_values,
        reference_distributions=reference_distributions,
    )
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
