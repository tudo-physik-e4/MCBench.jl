# Implementation of the PosteriorDB eight-schools example.
#
# The ready-to-use testcase constants are declared in
# `example_distributions.jl`, alongside the other built-in targets.


# ---------------------------------------------------------------------------
# Model data and target type
# ---------------------------------------------------------------------------

const _EIGHT_SCHOOLS_DIMENSION = 10
const _EIGHT_SCHOOLS_GROUPS = 8
const eight_schools_data = JSON.parsefile(joinpath(@__DIR__, "eight_schools.json"))

function _eight_schools_bounds(transformed::Bool)
    tau_bounds = transformed ? -100..100 : 1e-100..100
    NamedTupleDist(
        x=[index == 10 ? tau_bounds : -100..100 for index in 1:10],
    )
end

# Keep the original public binding available for existing callers.
const eight_schools_bounds = _eight_schools_bounds(false)

"""
Accept-reject implementation of the PosteriorDB eight-schools target.

The first eight dimensions contain the school effects `theta`, followed by the
population mean `mu` and scale `tau`. When `transformed=true`, the final stored
coordinate is `log(tau)` instead.
"""
struct EightSchoolsAcceptReject{D} <: Target
    data::D
    transformed::Bool
end

EightSchoolsAcceptReject(data) = EightSchoolsAcceptReject(data, false)
Base.length(::EightSchoolsAcceptReject) = _EIGHT_SCHOOLS_DIMENSION


# ---------------------------------------------------------------------------
# Exact accept-reject sampling
# ---------------------------------------------------------------------------

function _eight_schools_proposal(
    rng::AbstractRNG,
    y,
    covariance,
    maximum_log_likelihood;
    transformed::Bool,
)
    mu = rand(rng, Normal(0, 5))
    tau = abs(rand(rng, Cauchy(0, 5)))
    theta = rand(rng, Normal(mu, tau), _EIGHT_SCHOOLS_GROUPS)
    data_log_likelihood = logpdf(MvNormal(theta, covariance), y)

    # The proposal is the prior. Accepting according to the likelihood gives
    # exact posterior draws without requiring an MCMC warm-up.
    log(rand(rng)) < data_log_likelihood - maximum_log_likelihood || return nothing
    stored_tau = transformed ? log(tau) : tau
    (theta=theta, mu=mu, tau=stored_tau)
end

function _sample_eight_schools(
    rng::AbstractRNG,
    sample_count::Int,
    data;
    transformed::Bool,
)
    sample_count > 0 || throw(ArgumentError("sample count must be positive"))
    y = Float64.(data["y"])
    sigma = Float64.(data["sigma"])
    covariance = Diagonal(sigma .^ 2)
    maximum_log_likelihood = logpdf(MvNormal(zeros(8), covariance), zeros(8))
    accepted = NamedTuple[]
    sizehint!(accepted, sample_count)

    while length(accepted) < sample_count
        proposal = _eight_schools_proposal(
            rng,
            y,
            covariance,
            maximum_log_likelihood;
            transformed=transformed,
        )
        isnothing(proposal) || push!(accepted, proposal)
    end
    accepted
end

function Random.rand(
    rng::AbstractRNG,
    target::EightSchoolsAcceptReject,
    sample_count::Int,
)
    accepted = _sample_eight_schools(
        rng,
        sample_count,
        target.data;
        transformed=target.transformed,
    )
    values = Matrix{Float64}(undef, _EIGHT_SCHOOLS_DIMENSION, sample_count)

    for index in axes(values, 2)
        sample = accepted[index]
        values[:, index] .= (sample.theta..., sample.mu, sample.tau)
    end
    values
end

Random.rand(target::EightSchoolsAcceptReject, sample_count::Int) = rand(
    Random.default_rng(),
    target,
    sample_count,
)


# ---------------------------------------------------------------------------
# Posterior log density
# ---------------------------------------------------------------------------

function _logpdf_eight_schools(
    target::EightSchoolsAcceptReject,
    values::AbstractVector,
)
    length(values) == _EIGHT_SCHOOLS_DIMENSION || throw(DimensionMismatch(
        "expected a ten-dimensional eight-schools point",
    ))

    y = Float64.(target.data["y"])
    sigma = Float64.(target.data["sigma"])
    theta = values[1:_EIGHT_SCHOOLS_GROUPS]
    mu = values[9]
    stored_tau = values[10]
    tau = target.transformed ? exp(stored_tau) : stored_tau
    tau > 0 || return -Inf

    covariance = Diagonal(sigma .^ 2)
    data_log_likelihood = logpdf(MvNormal(theta, covariance), y)
    mu_log_prior = logpdf(Normal(0, 5), mu)

    # tau has a half-Cauchy prior because the generative sampler uses abs.
    tau_log_prior = log(2) + logpdf(Cauchy(0, 5), tau)
    target.transformed && (tau_log_prior += stored_tau) # tau = exp(stored_tau)

    theta_log_prior = sum(logpdf.(Ref(Normal(mu, tau)), theta))
    data_log_likelihood + mu_log_prior + tau_log_prior + theta_log_prior
end

function Distributions.logpdf(
    target::EightSchoolsAcceptReject,
    values::AbstractVector,
)
    _logpdf_eight_schools(target, values)
end

function Distributions.logpdf(
    target::EightSchoolsAcceptReject,
    values::AbstractMatrix,
)
    size(values, 1) == _EIGHT_SCHOOLS_DIMENSION || throw(DimensionMismatch(
        "expected a matrix with ten rows",
    ))
    [
        _logpdf_eight_schools(target, view(values, :, index))
        for index in axes(values, 2)
    ]
end


# ---------------------------------------------------------------------------
# Testcase factory
# ---------------------------------------------------------------------------

"""
    make_eight_schools_testcase(; transformed=false,
                                info="EightSchoolsAcceptReject")

Construct the PosteriorDB eight-schools testcase. With `transformed=true`, the
last coordinate stores `log(tau)` and therefore uses unconstrained bounds.
"""
function make_eight_schools_testcase(;
    transformed::Bool=false,
    info=transformed ?
        "EightSchoolsAcceptReject-Transformed" :
        "EightSchoolsAcceptReject",
)
    Testcases(
        EightSchoolsAcceptReject(eight_schools_data, transformed),
        _eight_schools_bounds(transformed),
        _EIGHT_SCHOOLS_DIMENSION,
        info,
    )
end
