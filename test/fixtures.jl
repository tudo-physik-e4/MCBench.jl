function standard_normal_testcase(dim::Int=2; info="Test-Normal-$(dim)D")
    distribution = dim == 1 ? Normal() : MvNormal(zeros(dim), Matrix{Float64}(I, dim, dim))
    bounds = NamedTupleDist(x=fill(-10..10, dim))
    MCBench.Testcases(distribution, bounds, dim, info)
end

sample_matrix(dsv) = hcat(BAT.unshaped.(dsv.v)...)

function write_lines(path, lines)
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    path
end

struct DeterministicTarget <: MCBench.Target
    dim::Int
end

Base.length(target::DeterministicTarget) = target.dim

function Base.rand(target::DeterministicTarget, n::Int)
    repeat(reshape(collect(1.0:target.dim), target.dim, 1), 1, n)
end

function Distributions.logpdf(::DeterministicTarget, values::AbstractMatrix)
    zeros(size(values, 2))
end

struct FailingSampler <: MCBench.SamplingAlgorithm
    info::String
end

function MCBench.sample(::MCBench.Testcases, ::FailingSampler; n_steps=10^5)
    error("intentional sampler failure")
end

mutable struct FixedEffectiveSampleSizeSampler <: MCBench.SamplingAlgorithm
    raw_draw_size::Int
    effective_draw_size::Float64
    draw_count::Int
    info::String
end

struct AutocorrelationESSSampler <: MCBench.SamplingAlgorithm
    info::String
end

function MCBench.sample(
    testcase::MCBench.Testcases,
    sampler::FixedEffectiveSampleSizeSampler;
    n_steps=10^5,
)
    sampler.draw_count += 1
    MCBench.sample(testcase; n_steps=sampler.raw_draw_size)
end

MCBench.get_effective_sample_size(
    ::DensitySampleVector,
    sampler::FixedEffectiveSampleSizeSampler,
) = sampler.effective_draw_size

struct SampleCountMetric{V<:Real,A} <: MCBench.TestMetric
    val::V
    info::A
end

SampleCountMetric() = SampleCountMetric(0, "SampleCount")

function MCBench.calc_metric(
    ::MCBench.AbstractTestcase,
    samples::DensitySampleVector,
    ::SampleCountMetric,
)
    [SampleCountMetric(length(samples), "SampleCount")]
end

struct TwoSampleCountMetric{V<:Real,A} <: MCBench.TwoSampleMetric
    val::V
    info::A
end

TwoSampleCountMetric() = TwoSampleCountMetric(0, "TwoSampleCount")

function MCBench.calc_metric(
    ::MCBench.AbstractTestcase,
    first::DensitySampleVector,
    second::DensitySampleVector,
    ::TwoSampleCountMetric,
)
    [TwoSampleCountMetric(1_000 * length(first) + length(second), "TwoSampleCount")]
end
