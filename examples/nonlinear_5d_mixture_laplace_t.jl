# Implementation of the Nonlinear 5D Mixture-Laplace-t target.
#
# The predefined easy and hard scenarios remain in `example_distributions.jl`
# so users can compare their parameter choices without reading these mechanics.


# ---------------------------------------------------------------------------
# Parameters and customization
# ---------------------------------------------------------------------------

"""
Parameters of the nonlinear five-dimensional target.

The fields are grouped by the conditional they control:

- `A`, `ω`, `σ1`, `b2`: x1 and the nonlinear x2 | x1 relation;
- `mode_sep1`, `mix_p1`: optional mixture for x1;
- `d3`, `γ3`, `ρ3`, `ν3`, `s3`: heavy-tailed x3 | x1,x2 relation;
- `b4_base`, `η4`: heteroskedastic x4 | x1 scale; and
- `b5`, `m5_base`, `m5_amp`, `ω5`, `κ`: gated x5 | x1,x2 mixture.
"""
struct NonlinearMixtureLaplaceTParams
    A::Float64
    ω::Float64
    σ1::Float64
    b2::Float64
    mode_sep1::Float64
    mix_p1::Float64
    d3::Float64
    γ3::Float64
    ρ3::Float64
    ν3::Float64
    s3::Float64
    b4_base::Float64
    η4::Float64
    b5::Float64
    m5_base::Float64
    m5_amp::Float64
    ω5::Float64
    κ::Float64
end

function NonlinearMixtureLaplaceTParams(;
    A=3.0,
    ω=1.6,
    σ1=3.0,
    b2=0.6,
    mode_sep1=0.0,
    mix_p1=0.3,
    d3=1.2,
    γ3=0.6,
    ρ3=0.8,
    ν3=4.0,
    s3=0.8,
    b4_base=0.25,
    η4=0.8,
    b5=0.2,
    m5_base=1.0,
    m5_amp=2.0,
    ω5=1.5,
    κ=10.0,
)
    σ1 > 0 || throw(ArgumentError("σ1 must be positive"))
    b2 > 0 || throw(ArgumentError("b2 must be positive"))
    0 < mix_p1 < 1 || throw(ArgumentError("mix_p1 must lie between zero and one"))
    ν3 > 2 || throw(ArgumentError("ν3 must exceed two so its variance is finite"))
    s3 > 0 || throw(ArgumentError("s3 must be positive"))
    b4_base > 0 || throw(ArgumentError("b4_base must be positive"))
    η4 >= 0 || throw(ArgumentError("η4 must be nonnegative"))
    b5 > 0 || throw(ArgumentError("b5 must be positive"))

    values = (
        A, ω, σ1, b2, mode_sep1, mix_p1, d3, γ3, ρ3, ν3, s3,
        b4_base, η4, b5, m5_base, m5_amp, ω5, κ,
    )
    NonlinearMixtureLaplaceTParams(Float64.(values)...)
end

function _nonlinear_parameter_values(params::NonlinearMixtureLaplaceTParams)
    names = fieldnames(NonlinearMixtureLaplaceTParams)
    NamedTuple{names}(Tuple(getfield(params, name) for name in names))
end

"""Return a copy of `params` with the supplied fields replaced."""
function withparams(params::NonlinearMixtureLaplaceTParams; kwargs...)
    values = merge(_nonlinear_parameter_values(params), (; kwargs...))
    NonlinearMixtureLaplaceTParams(; values...)
end


# ---------------------------------------------------------------------------
# Target type and generative sampler
# ---------------------------------------------------------------------------

"""
    NonlinearMixtureLaplaceT(params=NonlinearMixtureLaplaceTParams())

Normalized nonlinear target combining a Gaussian mixture, conditional Laplace
distributions, and a heavy-tailed Student-t conditional.
"""
struct NonlinearMixtureLaplaceT <: Target
    params::NonlinearMixtureLaplaceTParams
end

NonlinearMixtureLaplaceT() = NonlinearMixtureLaplaceT(
    NonlinearMixtureLaplaceTParams(),
)

Base.length(::NonlinearMixtureLaplaceT) = 5

function _nonlinear_logistic(value)
    value >= 0 ? inv(1 + exp(-value)) : exp(value) / (1 + exp(value))
end

function Random.rand(
    rng::AbstractRNG,
    target::NonlinearMixtureLaplaceT,
    sample_count::Int,
)
    sample_count > 0 || throw(ArgumentError("sample count must be positive"))
    p = target.params
    samples = Matrix{Float64}(undef, 5, sample_count)

    @inbounds for index in 1:sample_count
        # x1 is Gaussian in the easy case and a Gaussian mixture in the hard
        # case. A zero separation makes the mixture weight irrelevant.
        x1 = if iszero(p.mode_sep1)
            rand(rng, Normal(0.0, p.σ1))
        else
            component_mean = rand(rng) < p.mix_p1 ? p.mode_sep1 : -p.mode_sep1
            rand(rng, Normal(component_mean, p.σ1))
        end

        # x2 follows a sinusoidal ridge with Laplace noise.
        mean2 = p.A * sin(p.ω * x1)
        x2 = rand(rng, Laplace(mean2, p.b2))

        # x3 adds a second curve, residual coupling to x2, and Student-t tails.
        mean3 = p.d3 * sin(p.γ3 * x1) + p.ρ3 * (x2 - mean2)
        x3 = rand(rng, LocationScale(mean3, p.s3, TDist(p.ν3)))

        # The x4 scale grows quadratically with the magnitude of x1.
        scale4 = p.b4_base * (1 + p.η4 * (x1 / p.σ1)^2)
        x4 = rand(rng, Laplace(0.0, scale4))

        # x5 switches smoothly between two x1-dependent Laplace modes. The
        # x2 residual determines the mixture weight.
        mode5 = p.m5_base + p.m5_amp * sin(p.ω5 * x1)
        mixture_weight = _nonlinear_logistic(p.κ * (x2 - mean2))
        component_mean = rand(rng) < mixture_weight ? mode5 : -mode5
        x5 = rand(rng, Laplace(component_mean, p.b5))

        samples[:, index] .= (x1, x2, x3, x4, x5)
    end

    samples
end

Random.rand(target::NonlinearMixtureLaplaceT, sample_count::Int) = rand(
    Random.default_rng(),
    target,
    sample_count,
)


# ---------------------------------------------------------------------------
# Normalized log density
# ---------------------------------------------------------------------------

_nonlinear_softplus(value) = max(value, zero(value)) + log1p(exp(-abs(value)))

function _logpdf_nonlinear_mixture_laplace_t(
    target::NonlinearMixtureLaplaceT,
    values::AbstractVector,
)
    length(values) == 5 || throw(DimensionMismatch("expected a five-dimensional point"))
    p = target.params
    x1, x2, x3, x4, x5 = values

    # Use log-sum-exp for both mixtures to remain stable for separated modes.
    log_density = if iszero(p.mode_sep1)
        logpdf(Normal(0.0, p.σ1), x1)
    else
        positive = log(p.mix_p1) + logpdf(Normal(p.mode_sep1, p.σ1), x1)
        negative = log1p(-p.mix_p1) + logpdf(Normal(-p.mode_sep1, p.σ1), x1)
        max(positive, negative) + log1p(exp(-abs(positive - negative)))
    end

    mean2 = p.A * sin(p.ω * x1)
    log_density += logpdf(Laplace(mean2, p.b2), x2)

    mean3 = p.d3 * sin(p.γ3 * x1) + p.ρ3 * (x2 - mean2)
    log_density += logpdf(LocationScale(mean3, p.s3, TDist(p.ν3)), x3)

    scale4 = p.b4_base * (1 + p.η4 * (x1 / p.σ1)^2)
    log_density += logpdf(Laplace(0.0, scale4), x4)

    mode5 = p.m5_base + p.m5_amp * sin(p.ω5 * x1)
    gate = p.κ * (x2 - mean2)
    positive = -_nonlinear_softplus(-gate) + logpdf(Laplace(mode5, p.b5), x5)
    negative = -_nonlinear_softplus(gate) + logpdf(Laplace(-mode5, p.b5), x5)
    log_density + max(positive, negative) + log1p(exp(-abs(positive - negative)))
end

function Distributions.logpdf(
    target::NonlinearMixtureLaplaceT,
    values::AbstractVector,
)
    _logpdf_nonlinear_mixture_laplace_t(target, values)
end

function Distributions.logpdf(
    target::NonlinearMixtureLaplaceT,
    values::AbstractMatrix,
)
    size(values, 1) == 5 || throw(DimensionMismatch(
        "expected a matrix with five rows",
    ))
    [
        _logpdf_nonlinear_mixture_laplace_t(target, view(values, :, index))
        for index in axes(values, 2)
    ]
end


# ---------------------------------------------------------------------------
# Analytical reference values
# ---------------------------------------------------------------------------

function _normal_mixture_sine_moment(params, frequency)
    (2 * params.mix_p1 - 1) * sin(frequency * params.mode_sep1) *
    exp(-0.5 * frequency^2 * params.σ1^2)
end

function _normal_mixture_sine_second_moment(params, frequency)
    0.5 * (
        1 - cos(2 * frequency * params.mode_sep1) *
        exp(-2 * frequency^2 * params.σ1^2)
    )
end

function _nonlinear_reference_values(params::NonlinearMixtureLaplaceTParams)
    mean1 = (2 * params.mix_p1 - 1) * params.mode_sep1
    variance1 = params.σ1^2 + params.mode_sep1^2 - mean1^2

    sine2_mean = _normal_mixture_sine_moment(params, params.ω)
    sine2_variance = _normal_mixture_sine_second_moment(params, params.ω) -
        sine2_mean^2
    mean2 = params.A * sine2_mean
    variance2 = params.A^2 * sine2_variance + 2 * params.b2^2

    sine3_mean = _normal_mixture_sine_moment(params, params.γ3)
    sine3_variance = _normal_mixture_sine_second_moment(params, params.γ3) -
        sine3_mean^2
    mean3 = params.d3 * sine3_mean
    variance3 = params.d3^2 * sine3_variance + 2 * params.ρ3^2 * params.b2^2 +
        params.s3^2 * params.ν3 / (params.ν3 - 2)

    second_moment1 = params.σ1^2 + params.mode_sep1^2
    fourth_moment1 = 3 * params.σ1^4 +
        6 * params.σ1^2 * params.mode_sep1^2 +
        params.mode_sep1^4
    variance4 = 2 * params.b4_base^2 * (
        1 + 2 * params.η4 * second_moment1 / params.σ1^2 +
        params.η4^2 * fourth_moment1 / params.σ1^4
    )

    sine5_mean = _normal_mixture_sine_moment(params, params.ω5)
    sine5_second = _normal_mixture_sine_second_moment(params, params.ω5)
    variance5 = params.m5_base^2 +
        2 * params.m5_base * params.m5_amp * sine5_mean +
        params.m5_amp^2 * sine5_second + 2 * params.b5^2

    (
        marginal_mean=[mean1, mean2, mean3, 0.0, 0.0],
        marginal_variance=[variance1, variance2, variance3, variance4, variance5],
        wasserstein_1d=zeros(5),
        sliced_wasserstein_distance=0.0,
        maximum_mean_discrepancy=0.0,
    )
end


# ---------------------------------------------------------------------------
# Testcase factory
# ---------------------------------------------------------------------------

"""
    nonlinear_5d_mixture_laplace_t(params=NonlinearMixtureLaplaceTParams();
                                   info="Nonlinear-5D-Mixture-Laplace-t")

Build a five-dimensional MCBench testcase from a nonlinear target parameter
set. Use the predefined easy or hard testcase unless a custom scenario is
needed.
"""
function nonlinear_5d_mixture_laplace_t(
    params::NonlinearMixtureLaplaceTParams=NonlinearMixtureLaplaceTParams();
    info="Nonlinear-5D-Mixture-Laplace-t",
)
    target = NonlinearMixtureLaplaceT(params)
    Testcases(
        target,
        NamedTupleDist(x=fill(-100..100, 5)),
        5,
        info;
        reference_values=_nonlinear_reference_values(params),
    )
end
