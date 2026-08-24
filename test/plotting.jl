@testset "Plot generation" begin
    mktempdir() do dir
        cd(dir) do
            mkpath("teststatistics")
            mkpath("teststatistics_sampler")
            distribution = Normal()
            bounds = NamedTupleDist(x=[-5..5])
            testcase = MCBench.Testcases(
                distribution,
                bounds,
                1,
                "Plot-Test";
                reference_values=(
                    marginal_mean=0.05,
                    marginal_quantiles=metric -> collect(metric.probabilities),
                ),
            )
            metric = MCBench.marginal_mean()
            quantile_metric = MCBench.marginal_quantiles()
            source = MCBench.make_dsv(collect(-2.0:2.0))
            sampler = MCBench.DsvSampler([source]; info="Compared")

            write_lines(
                joinpath("teststatistics", "Plot-Test-Mean.txt"),
                ["[-0.2]", "[-0.1]", "[0.0]", "[0.1]", "[0.2]"],
            )
            write_lines(
                joinpath("teststatistics_sampler", "Plot-Test-Mean-Compared.txt"),
                ["[-0.1]", "[0.0]", "[0.1]", "[0.2]", "[0.3]"],
            )
            write_lines(
                joinpath("teststatistics", "Plot-Test-Quantiles-50-90-99.txt"),
                ["[0.0, 0.9, 0.99]", "[0.1, 1.0, 1.1]", "[-0.1, 0.8, 0.9]"],
            )
            write_lines(
                joinpath(
                    "teststatistics_sampler",
                    "Plot-Test-Quantiles-50-90-99-Compared.txt",
                ),
                ["[0.1, 1.0, 1.1]", "[0.2, 1.1, 1.2]", "[0.0, 0.9, 1.0]"],
            )

            iid_paths = MCBench.plot_teststatistic(testcase, metric)
            iid_plot = joinpath("teststatistics", "Plot-Test-Mean-x1.pdf")
            @test iid_paths == [iid_plot]
            @test isfile(iid_plot)
            @test filesize(iid_plot) > 0
            @test length(MCBench.Plots.current().series_list) == 2

            comparison_paths = MCBench.plot_teststatistic(testcase, metric, sampler; nbins=3)
            comparison_plot = joinpath("Plot-Test", "Plot-Test-Mean-Compared-x1.pdf")
            @test comparison_paths == [comparison_plot]
            @test isfile(comparison_plot)
            @test filesize(comparison_plot) > 0
            @test length(MCBench.Plots.current().series_list) == 3

            rm(comparison_plot)
            unsaved_plots = MCBench.plot_teststatistic(
                testcase,
                metric,
                sampler;
                nbins=3,
                save_plots=false,
            )
            @test length(unsaved_plots) == 1
            @test unsaved_plots[1] isa MCBench.Plots.Plot
            @test !isfile(comparison_plot)

            MCBench.plot_teststatistic(
                testcase,
                metric,
                sampler;
                nbins=3,
                show_reference=false,
            )
            @test length(MCBench.Plots.current().series_list) == 2

            MCBench.plot_metrics(testcase, [metric], sampler; names=["x"])
            metrics_pdf = joinpath("Plot-Test", "Plot-Test-Compared-metrics.pdf")
            metrics_png = joinpath("Plot-Test", "Plot-Test-Compared-metrics.png")
            @test isfile(metrics_pdf)
            @test isfile(metrics_png)
            @test filesize(metrics_pdf) > 0
            @test filesize(metrics_png) > 0

            rm(metrics_pdf)
            rm(metrics_png)
            unsaved_overview = MCBench.plot_metrics(
                testcase,
                [metric],
                sampler;
                names=["x"],
                save_plots=false,
            )
            @test unsaved_overview isa MCBench.Plots.Plot
            expected_iid_normalized_value = 0.1 / std([-0.2, -0.1, 0.0, 0.1, 0.2])
            @test any(
                series -> any(x -> x ≈ expected_iid_normalized_value, series[:x]),
                unsaved_overview.series_list,
            )
            @test !isfile(metrics_pdf)
            @test !isfile(metrics_png)

            quantile_overview = MCBench.plot_metrics(
                testcase,
                [quantile_metric],
                sampler;
                names=["x"],
                save_plots=false,
            )
            @test quantile_overview isa MCBench.Plots.Plot

            quantile_reference_overview = MCBench.plot_reference_metrics(
                testcase,
                [quantile_metric],
                sampler;
                names=["x"],
                save_plots=false,
            )
            @test quantile_reference_overview isa MCBench.Plots.Plot

            # Metrics without a stored reference are silently omitted.
            reference_paths = MCBench.plot_reference_metrics(
                testcase,
                [MCBench.marginal_variance(), metric],
                sampler;
                names=["x"],
            )
            reference_pdf = joinpath(
                "Plot-Test",
                "Plot-Test-Compared-reference-metrics.pdf",
            )
            reference_png = joinpath(
                "Plot-Test",
                "Plot-Test-Compared-reference-metrics.png",
            )
            @test reference_paths == [reference_pdf, reference_png]
            @test all(isfile, reference_paths)
            @test all(path -> filesize(path) > 0, reference_paths)

            unsaved_reference_overview = MCBench.plot_reference_metrics(
                testcase,
                [metric],
                sampler;
                names=["x"],
                save_plots=false,
            )
            @test unsaved_reference_overview isa MCBench.Plots.Plot
            marker_series = only(filter(
                series -> series[:seriestype] == :scatter,
                unsaved_reference_overview.series_list,
            ))
            @test any(x -> x ≈ 0.05, marker_series[:x])
            labels = string.(getindex.(unsaved_reference_overview.series_list, :label))
            @test all(
                label -> label in labels,
                ["1σ region", "2σ region", "3σ region"],
            )

            missing_reference = standard_normal_testcase(1; info="Missing-Reference")
            @test_throws ArgumentError MCBench.plot_reference_metrics(
                missing_reference,
                [metric],
                sampler;
            )

            MCBench.Plots.closeall()
        end
    end
end
