import Plots

@testset "End-to-end benchmark workflow" begin
    mktempdir() do output_dir
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

        rng = MersenneTwister(88)
        candidate_distribution = MvNormal(
            [0.3, -0.2],
            Matrix{Float64}(I, 2, 2),
        )
        candidate_values = rand(rng, candidate_distribution, 30)
        candidate_samples = MCBench.make_dsv(
            candidate_values,
            logpdf(target, candidate_values),
        )
        sampler = MCBench.DsvSampler(
            [candidate_samples];
            info="Shifted-Sampler",
        )

        result = cd(output_dir) do
            mkpath("teststatistics")
            mkpath("teststatistics_sampler")

            Random.seed!(88)
            MCBench.build_teststatistic(
                testcase,
                metrics;
                n=3,
                n_steps=30,
                n_samples=30,
                par=false,
                clean=true,
                use_sampler=false,
            )
            MCBench.build_teststatistic(
                testcase,
                metrics;
                s=sampler,
                n=3,
                n_steps=30,
                n_samples=30,
                par=false,
                clean=true,
                use_sampler=false,
            )

            iid_mean = MCBench.read_teststatistic(testcase, metrics[1])
            sampler_mean = MCBench.read_teststatistic(
                testcase,
                metrics[1],
                sampler,
            )

            plot_dir = joinpath(output_dir, testcase.info)
            mkpath(plot_dir)
            plot_files = String[]

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
                    Plots.savefig(metric_plot, plot_path)
                    push!(plot_files, plot_path)
                end
            end

            overview_plot = MCBench.plot_metrics(
                testcase,
                metrics,
                sampler;
                names=["x₁", "x₂"],
                save_plots=false,
            )
            overview_pdf = joinpath(plot_dir, "metric-overview.pdf")
            overview_png = joinpath(plot_dir, "metric-overview.png")
            Plots.savefig(overview_plot, overview_pdf)
            Plots.savefig(overview_plot, overview_png)
            append!(plot_files, (overview_pdf, overview_png))

            (
                iid_mean=iid_mean,
                sampler_mean=sampler_mean,
                plot_files=plot_files,
            )
        end

        @test size(result.iid_mean) == (2, 3)
        @test size(result.sampler_mean) == (2, 3)
        @test length(result.plot_files) == 6
        @test any(contains("Variance"), result.plot_files)
        @test all(isfile, result.plot_files)
        @test all(path -> filesize(path) > 0, result.plot_files)
        Plots.closeall()
    end
end

