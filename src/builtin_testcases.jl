# Built-in testcase catalog
# =========================
#
# This file lists the distributions shipped with MCBench. Most users only need
# to select one of the testcase constants below, for example:
#
#     testcase = MCBench.normal_3d_uncorrelated
#     testcase = MCBench.nonlinear_5d_mixture_laplace_t_hard
#
# Each testcase contains a sampleable target, finite BAT bounds, a display name,
# and analytical references where they are available.
#
# Quick guide
# -----------
# - `normal_*_uncorrelated`: simple baseline targets.
# - `normal_*_weakly_correlated`: mild linear dependence.
# - `normal_*_strongly_correlated`: difficult linear dependence.
# - `normal_*_multimodal_*`: separated Gaussian modes.
# - `cauchy_1d`: heavy tails without finite mean or variance.
# - `nonlinear_5d_mixture_laplace_t_*`: nonlinear dependence, mixtures,
#   heteroskedasticity, and heavy tails.
# - `eight_schools_testcase*`: a realistic hierarchical posterior, with or
#   without a log transformation of its scale parameter.


# ---------------------------------------------------------------------------
# Shared construction helpers
# ---------------------------------------------------------------------------

# SWD and MMD are deliberately omitted: their ideal score is not a property of
# the target distribution itself.
function _normal_reference_values(dim::Int)
    (
        marginal_mean=zeros(dim),
        marginal_variance=ones(dim),
        global_mode=zeros(dim),
        marginal_mode=zeros(dim),
        marginal_skewness=zeros(dim),
        marginal_kurtosis=zeros(dim),
        marginal_quantiles=metric -> [
            Distributions.quantile(Normal(), probability)
            for probability in metric.probabilities
            for _ in 1:dim
        ],
        wasserstein_1d=zeros(dim),
    )
end

# For a Gaussian, the squared Mahalanobis distance follows ChiSquared(dim).
_normal_reference_distributions(distribution) = (
    squared_mahalanobis=mahalanobis_reference(distribution),
)

function _moment_reference_values(distribution)
    distribution_mean = Distributions.mean(distribution)
    dim = distribution_mean isa Real ? 1 : length(distribution_mean)
    (
        marginal_mean=distribution_mean,
        marginal_variance=Distributions.var(distribution),
        wasserstein_1d=zeros(dim),
    )
end

function _bounded_testcase(
    distribution,
    dim::Int,
    info::String;
    interval=-10..10,
    reference_values=(;),
    reference_distributions=(;),
)
    bounds = NamedTupleDist(x=fill(interval, dim))
    Testcases(
        distribution,
        bounds,
        dim,
        info;
        reference_values=reference_values,
        reference_distributions=reference_distributions,
    )
end


# ---------------------------------------------------------------------------
# 1. Gaussian baseline targets
# ---------------------------------------------------------------------------

function _normal_testcase(dim::Int, info::String; correlation=0.0)
    distribution = if dim == 1
        Normal()
    else
        covariance = fill(correlation, dim, dim)
        covariance[diagind(covariance)] .= 1.0
        MvNormal(zeros(dim), covariance)
    end

    _bounded_testcase(
        distribution,
        dim,
        info;
        reference_values=_normal_reference_values(dim),
        reference_distributions=_normal_reference_distributions(distribution),
    )
end

# Independent standard normals provide simple baseline cases.
const normal_1d_uncorrelated = _normal_testcase(1, "Normal-1D-Uncorrelated")
const normal_2d_uncorrelated = _normal_testcase(2, "Normal-2D-Uncorrelated")
"""Three-dimensional independent standard-normal testcase."""
const normal_3d_uncorrelated = _normal_testcase(3, "Normal-3D-Uncorrelated")
const normal_10d_uncorrelated = _normal_testcase(10, "Normal-10D-Uncorrelated")
const normal_100d_uncorrelated = _normal_testcase(100, "Normal-100D-Uncorrelated")

# Every pair of dimensions has the correlation shown in the testcase name.
const normal_2d_weakly_correlated = _normal_testcase(
    2,
    "Normal-2D-Weakly-Correlated";
    correlation=0.3,
)
const normal_2d_strongly_correlated = _normal_testcase(
    2,
    "Normal-2D-Strongly-Correlated";
    correlation=0.9,
)
const normal_10d_weakly_correlated = _normal_testcase(
    10,
    "Normal-10D-Weakly-Correlated";
    correlation=0.2,
)
const normal_10d_strongly_correlated = _normal_testcase(
    10,
    "Normal-10D-Strongly-Correlated";
    correlation=0.9,
)
const normal_100d_weakly_correlated = _normal_testcase(
    100,
    "Normal-100D-Weakly-Correlated";
    correlation=0.2,
)
const normal_100d_strongly_correlated = _normal_testcase(
    100,
    "Normal-100D-Strongly-Correlated";
    correlation=0.9,
)


# ---------------------------------------------------------------------------
# 2. Gaussian mixture targets
# ---------------------------------------------------------------------------

function _univariate_mixture_testcase(
    separation,
    weights,
    info;
    interval=-10..10,
)
    distribution = MixtureModel(
        [Normal(separation, 1), Normal(-separation, 1)],
        weights,
    )
    _bounded_testcase(
        distribution,
        1,
        info;
        interval=interval,
        reference_values=_moment_reference_values(distribution),
    )
end

# The 1-to-3 variants have unequal mode weights.
const normal_1d_multimodal_4std = _univariate_mixture_testcase(
    2,
    [0.5, 0.5],
    "Normal-1D-Multimodal-4std",
)
const normal_1d_multimodal_20std = _univariate_mixture_testcase(
    10,
    [0.5, 0.5],
    "Normal-1D-Multimodal-20std";
    interval=-20..20,
)
const normal_1d_multimodal_4std_1to3 = _univariate_mixture_testcase(
    2,
    [0.25, 0.75],
    "Normal-1D-Multimodal-4std-1to3",
)
const normal_1d_multimodal_20std_1to3 = _univariate_mixture_testcase(
    10,
    [0.25, 0.75],
    "Normal-1D-Multimodal-20std-1to3";
    interval=-20..20,
)

function _multivariate_mixture_testcase(dim::Int, info::String)
    separation = 5
    covariance = fill(0.9, dim, dim)
    covariance[diagind(covariance)] .= 1.0
    components = [
        MvNormal(fill(separation, dim), covariance),
        MvNormal(fill(-separation, dim), covariance),
    ]
    distribution = MixtureModel(components, [0.25, 0.75])

    _bounded_testcase(
        distribution,
        dim,
        info;
        interval=-100..100,
        reference_values=_moment_reference_values(distribution),
    )
end

# These mixtures combine separated, unequal modes with strong correlation.
const normal_3d_multimodal_10std = _multivariate_mixture_testcase(
    3,
    "Normal-3D-Multimodal-10std",
)
const normal_10d_multimodal_10std = _multivariate_mixture_testcase(
    10,
    "Normal-10D-Multimodal-10std",
)


# ---------------------------------------------------------------------------
# 3. Heavy-tailed target
# ---------------------------------------------------------------------------

# Cauchy has no finite mean or variance, so those reference values are
# intentionally absent.
const cauchy_1d = _bounded_testcase(Cauchy(), 1, "Cauchy-1D")


# ---------------------------------------------------------------------------
# 4. Nonlinear 5D Mixture-Laplace-t targets
# ---------------------------------------------------------------------------

# The implementation lives in a separate file to keep this catalog easy to
# scan. The target combines:
#
# - an optional two-component Gaussian mixture for x1;
# - a nonlinear Laplace conditional for x2;
# - a curved, heavy-tailed Student-t conditional for x3;
# - a heteroskedastic Laplace conditional for x4; and
# - a smoothly gated Laplace mixture for x5.
include("nonlinear_5d_mixture_laplace_t.jl")

# Easy scenario: x1 is unimodal, the x1-x2 curve varies slowly, and the
# remaining dependencies and heteroskedasticity are mild.
"""Parameter set used by [`nonlinear_5d_mixture_laplace_t_easy`](@ref)."""
const nonlinear_5d_mixture_laplace_t_easy_params = NonlinearMixtureLaplaceTParams(
    A=2.0,
    ω=0.2,
    σ1=3.0,
    b2=0.6,
    mode_sep1=0.0,
    mix_p1=0.01,
    d3=1.2,
    γ3=0.6,
    ρ3=0.3,
    ν3=4.0,
    s3=0.8,
    b4_base=0.6,
    η4=0.05,
    b5=0.1,
    m5_base=0.0,
    m5_amp=0.1,
    ω5=0.5,
    κ=0.8,
)

# Hard scenario: x1 has two widely separated, unequally weighted modes. Its
# faster x1-x2 oscillation is layered on top of the stronger default nonlinear
# dependencies, heteroskedasticity, and x5 mixture gate.
"""Parameter set used by [`nonlinear_5d_mixture_laplace_t_hard`](@ref)."""
const nonlinear_5d_mixture_laplace_t_hard_params = withparams(
    NonlinearMixtureLaplaceTParams();
    mode_sep1=8.0,
    mix_p1=0.3,
    ω=0.7,
)

"""Ready-to-use testcase for the easier, unimodal nonlinear 5D target."""
const nonlinear_5d_mixture_laplace_t_easy = nonlinear_5d_mixture_laplace_t(
    nonlinear_5d_mixture_laplace_t_easy_params;
    info="Nonlinear-5D-Mixture-Laplace-t-Easy",
)

"""Ready-to-use testcase for the harder, multimodal nonlinear 5D target."""
const nonlinear_5d_mixture_laplace_t_hard = nonlinear_5d_mixture_laplace_t(
    nonlinear_5d_mixture_laplace_t_hard_params;
    info="Nonlinear-5D-Mixture-Laplace-t-Hard",
)

export NonlinearMixtureLaplaceTParams, NonlinearMixtureLaplaceT, withparams
export nonlinear_5d_mixture_laplace_t
export nonlinear_5d_mixture_laplace_t_easy, nonlinear_5d_mixture_laplace_t_hard
export nonlinear_5d_mixture_laplace_t_easy_params
export nonlinear_5d_mixture_laplace_t_hard_params


# ---------------------------------------------------------------------------
# 5. PosteriorDB: eight schools
# ---------------------------------------------------------------------------

# The eight-schools model is a standard hierarchical Bayesian benchmark. The
# transformed version stores log(tau), which removes the positive boundary on
# the population-scale parameter and is often easier for MCMC samplers.
#
# Model data, sampling, and log-density details live in `eight_schools.jl`.
include("eight_schools.jl")

const eight_schools_testcase = make_eight_schools_testcase(
    info="EightSchoolsAcceptReject",
)

const eight_schools_testcase_trafo = make_eight_schools_testcase(
    transformed=true,
    info="EightSchoolsAcceptReject-Transformed",
)

export EightSchoolsAcceptReject, make_eight_schools_testcase
export eight_schools_testcase, eight_schools_testcase_trafo
