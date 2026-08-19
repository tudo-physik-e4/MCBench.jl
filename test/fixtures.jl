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
