module BasicMetricsExample

using Random
using LinearAlgebra
using Distributions
using IntervalSets
using ValueShapes

import MCBench

"""
    main(; n_samples=2_000, seed=1234, verbose=true)

Run a small, in-memory MCBench comparison. The candidate sample is deliberately
shifted so that the reported metrics are easy to interpret.
"""
function main(; n_samples::Int=2_000, seed::Int=1234, verbose::Bool=true)
    rng = MersenneTwister(seed)
    target = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
    bounds = NamedTupleDist(x=fill(-6..6, 2))
    testcase = MCBench.Testcases(
        target,
        bounds,
        2,
        "Basic-Metrics-Normal";
        reference_values=(
            marginal_mean=zeros(2),
            marginal_variance=ones(2),
        ),
    )

    iid_values = rand(rng, target, n_samples)
    candidate_distribution = MvNormal([0.25, -0.15], Matrix{Float64}(I, 2, 2))
    candidate_values = rand(rng, candidate_distribution, n_samples)

    iid_samples = MCBench.make_dsv(iid_values, logpdf(target, iid_values))
    candidate_samples = MCBench.make_dsv(candidate_values, logpdf(target, candidate_values))

    means = MCBench.calc_metric(testcase, candidate_samples, MCBench.marginal_mean())
    variances = MCBench.calc_metric(testcase, candidate_samples, MCBench.marginal_variance())
    wasserstein = MCBench.get_sliced_wasserstein_distance(
        iid_samples,
        candidate_samples;
        L=32,
        parallel=false,
    )
    mmd = MCBench.get_mmd(iid_samples, candidate_samples)

    result = (
        mean=[metric.val for metric in means],
        variance=[metric.val for metric in variances],
        wasserstein=wasserstein,
        mmd=mmd,
        reference_mean=MCBench.reference_values(testcase, MCBench.marginal_mean()),
    )

    if verbose
        println("Candidate marginal means: ", result.mean)
        println("Candidate marginal variances: ", result.variance)
        println("Sliced Wasserstein distance: ", result.wasserstein)
        println("Maximum mean discrepancy: ", result.mmd)
    end

    result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
