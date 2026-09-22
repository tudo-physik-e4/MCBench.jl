"""
    get_nammd(s1::DensitySampleVector, s2::DensitySampleVector; g=0, N=0)
    get_nammd(x::AbstractArray, y::AbstractArray; g=0)

Compute the Norm-Adaptive Maximum Mean Discrepancy (NAMMD) between two samples,
following Zhou et al., "A Kernel Distributional Closeness Testing", ICML 2026:

    NAMMD(P, Q; kappa) = MMD^2(P, Q; kappa) / (4K - ||mu_P||^2 - ||mu_Q||^2)

This reuses the existing `get_mmd`/`kernelsum`/`GaussianKernel` machinery from
`mmd.jl`: the denominator terms ||mu_P||^2, ||mu_Q||^2 are exactly the
within-sample U-statistic estimates already computed inside `kernelsum`.
"""
function get_nammd(s1::DensitySampleVector, s2::DensitySampleVector; g=0, N=0)
    s1, s2 = prepare_twosample_dsv(s1, s2, N=N)
    x = _sample_value_matrix(s1)
    y = _sample_value_matrix(s2)
    get_nammd(x, y, g=g)
end

function get_nammd(x::AbstractArray, y::AbstractArray; g=0)
    γ = g != 0 ? g : compute_bandwidth(x, y)
    k = GaussianKernel(γ)

    xx = kernelsum(k, x, pairwisel2)   # U-statistic estimate of ||mu_P||^2_Hk
    yy = kernelsum(k, y, pairwisel2)   # U-statistic estimate of ||mu_Q||^2_Hk
    xy = kernelsum(k, x, y, pairwisel2)
    mmd_val = xx + yy - 2xy

    K = k(0.0)                        # kappa(z,z) upper bound; = 1 for GaussianKernel
    denom = 4K - xx - yy
    return mmd_val / denom
end

"""
    struct norm_adaptive_mmd{V<:Real,I<:Int,A<:Any,P<:Any} <: TwoSampleMetric

    # Fields
    - `val::V`: The value of the metric.
    - `N::I`: The number of points used to calculate NAMMD.
    - `info::A`: Information about the metric.
    - `proc::P`: The processor information for the metric.

    # Constructors
    - `norm_adaptive_mmd(; fields...)`
    - `norm_adaptive_mmd(val::Real)`

    A struct for the Norm-Adaptive MMD (NAMMD) of the samples as a metric.
    Unlike raw MMD, NAMMD rescales the discrepancy by the RKHS norms of the
    two distributions, which improves finite-sample distinguishability when
    distribution pairs share the same MMD value but different norms.
"""
struct norm_adaptive_mmd{
    V<:Real,
    I<:Int,
    A<:Any,
    P<:Any
} <: TwoSampleMetric
    val::V
    N::I
    info::A
    proc::P
end
export norm_adaptive_mmd

function norm_adaptive_mmd() norm_adaptive_mmd(0.0, 10^4, "NormAdaptiveMMD", wd_pint()) end
function norm_adaptive_mmd(val::V) where {V<:Real} norm_adaptive_mmd(val, 10^4, "NormAdaptiveMMD", "") end
function norm_adaptive_mmd(val::V, N::Int) where {V<:Real} norm_adaptive_mmd(val, N, "NormAdaptiveMMD", wd_pint()) end

function calc_metric(t::AT, s1::DensitySampleVector, s2::DensitySampleVector, m::norm_adaptive_mmd) where {AT<:AbstractTestcase}
    return [(norm_adaptive_mmd(get_nammd(s1, s2, N=m.N)))]
end

export get_nammd
