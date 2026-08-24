"""Base type for Markov-chain Monte Carlo sampling algorithms."""
abstract type MCMCSamplingAlgorithm <: SamplingAlgorithm end

"""Configuration wrapper for BAT's Metropolis-Hastings sampler."""
struct BATMH{SA<:BAT.AbstractSamplingAlgorithm,A} <: MCMCSamplingAlgorithm
    sampler::SA
    info::A
end

function BATMH(; n_steps::Int=100_000, nchains::Int=10)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    algorithm = _bat_mh_algorithm(n_steps, nchains)
    BATMH(algorithm, "BAT-MH")
end

# BAT 4 renamed the MCMC configuration types. Keep the fallback so MCBench's
# declared BAT 3 compatibility remains usable without warnings on BAT 4.
function _bat_mh_algorithm(n_steps::Int, nchains::Int)
    if isdefined(BAT, :TransformedMCMC) && isdefined(BAT, :RandomWalk)
        return BAT.TransformedMCMC(
            proposal=BAT.RandomWalk(),
            nsteps=n_steps,
            nchains=nchains,
        )
    end

    BAT.MCMCSampling(
        mcalg=BAT.MetropolisHastings(),
        nsteps=n_steps,
        nchains=nchains,
    )
end

"""Build the bounded BAT posterior associated with a testcase."""
build_bat_posterior(testcase::Testcases) =
    BAT.PosteriorMeasure(testcase, testcase.bounds)

# Backward-compatible spelling retained for existing callers.
buildBATPosterior(testcase::Testcases) = build_bat_posterior(testcase)

function DensityInterface.logdensityof(testcase::Testcases, point)
    result = logpdf(testcase.f, point.x)
    result isa Real ? result : first(result)
end

function sample(
    testcase::Testcases,
    sampler::BATMH;
    n_steps::Int=100_000,
    nchains::Int=10,
)
    algorithm = if n_steps == 100_000 && nchains == 10
        sampler.sampler
    else
        BATMH(n_steps=n_steps, nchains=nchains).sampler
    end
    bat_sample(build_bat_posterior(testcase), algorithm).result
end

export MCMCSamplingAlgorithm, BATMH, build_bat_posterior, buildBATPosterior
