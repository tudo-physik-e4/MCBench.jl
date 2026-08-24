module EndToEndBenchmarkExample

using Random
using LinearAlgebra
using Distributions
using IntervalSets
using Plots
using ValueShapes

import MCBench

"""
    main(; output_dir=mktempdir(), n_repetitions=20, n_samples=500,
           seed=4321, verbose=true)

Run the complete lightweight benchmark workflow: build IID and candidate test
statistics, read them back, and create individual and overview plots.
"""
function main(;
    output_dir::AbstractString=mktempdir(),
    n_repetitions::Int=20,
    n_samples::Int=500,
    seed::Int=4321,
    verbose::Bool=true,
)
    mkpath(output_dir)
    output_dir = abspath(output_dir)
    rng = MersenneTwister(seed)

    target = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
    bounds = NamedTupleDist(x=fill(-6..6, 2))
    testcase = MCBench.Testcases(
        target,
        bounds,
        2,
        "End-to-End-Normal";
        reference_values=(
            marginal_mean=zeros(2),
            marginal_variance=ones(2),
        ),
    )
    metrics = MCBench.TestMetric[
        MCBench.marginal_mean(),
        MCBench.marginal_variance(),
    ]

    candidate_distribution = MvNormal([0.3, -0.2], Matrix{Float64}(I, 2, 2))
    candidate_values = rand(rng, candidate_distribution, n_samples)
    candidate_samples = MCBench.make_dsv(
        candidate_values,
        logpdf(target, candidate_values),
    )
    sampler = MCBench.DsvSampler([candidate_samples]; info="Shifted-Sampler")

    result = cd(output_dir) do
        mkpath("teststatistics")
        mkpath("teststatistics_sampler")

        Random.seed!(seed)
        MCBench.build_teststatistic(
            testcase,
            metrics;
            n=n_repetitions,
            n_steps=n_samples,
            n_samples=n_samples,
            par=false,
            clean=true,
            use_sampler=false,
        )
        MCBench.build_teststatistic(
            testcase,
            metrics;
            s=sampler,
            n=n_repetitions,
            n_steps=n_samples,
            n_samples=n_samples,
            par=false,
            clean=true,
            use_sampler=false,
        )

        iid_mean = MCBench.read_teststatistic(testcase, metrics[1])
        sampler_mean = MCBench.read_teststatistic(testcase, metrics[1], sampler)

        plot_dir = joinpath(output_dir, testcase.info)
        mkpath(plot_dir)
        plot_files = String[]

        # Build and save an individual comparison plot for every metric and
        # every parameter. Keeping these as separate lines makes it easy to
        # customize a plot before writing it.
        for metric in metrics
            metric_plots = MCBench.plot_teststatistic(
                testcase,
                metric,
                sampler;
                nbins=8,
                save_plots=false,
            )

            for (dimension, metric_plot) in enumerate(metric_plots)
                plot_path = joinpath(
                    plot_dir,
                    "$(testcase.info)-$(metric.info)-$(sampler.info)-x$dimension.pdf",
                )
                savefig(metric_plot, plot_path)
                push!(plot_files, plot_path)
            end
        end

        # The overview contains all metrics in one normalized plot.
        overview_plot = MCBench.plot_metrics(
            testcase,
            metrics,
            sampler;
            names=["x₁", "x₂"],
            save_plots=false,
        )

        overview_pdf = joinpath(
            plot_dir,
            "$(testcase.info)-$(sampler.info)-metrics.pdf",
        )
        savefig(overview_plot, overview_pdf)
        push!(plot_files, overview_pdf)

        overview_png = joinpath(
            plot_dir,
            "$(testcase.info)-$(sampler.info)-metrics.png",
        )
        savefig(overview_plot, overview_png)
        push!(plot_files, overview_png)

        (iid_mean=iid_mean, sampler_mean=sampler_mean, plot_files=plot_files)
    end

    closeall()
    verbose && println("Benchmark outputs written to ", output_dir)
    result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(output_dir=joinpath(@__DIR__, "output"))
end

end # module
