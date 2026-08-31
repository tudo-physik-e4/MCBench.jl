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
                    marginal_quantiles=metric -> Union{Missing,Float64}[
                        metric.probabilities[1],
                        missing,
                        missing,
                    ],
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
            expected_plot_ks = MCBench.ks_test(
                [-0.2, -0.1, 0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1, 0.2, 0.3],
            )
            plot_title = string(unsaved_plots[1][1][:title])
            @test contains(plot_title, "KS test:")
            @test contains(
                plot_title,
                "D = $(round(expected_plot_ks.statistic; sigdigits=4))",
            )
            @test contains(
                plot_title,
                "p = $(round(expected_plot_ks.pvalue; sigdigits=4))",
            )
            expected_reference_test = MCBench.reference_value_test(
                [-0.1, 0.0, 0.1, 0.2, 0.3],
                0.05,
            )
            @test contains(plot_title, "Reference t-test (Compared):")
            @test contains(
                plot_title,
                "t = $(round(expected_reference_test.statistic; sigdigits=4))",
            )
            @test contains(
                plot_title,
                "p = $(round(expected_reference_test.pvalue; sigdigits=4))",
            )
            @test !isfile(comparison_plot)

            MCBench.plot_teststatistic(
                testcase,
                metric,
                sampler;
                nbins=3,
                show_reference=false,
                show_reference_test=false,
                show_ks_test=false,
            )
            @test length(MCBench.Plots.current().series_list) == 2
            @test !contains(
                string(MCBench.Plots.current()[1][:title]),
                "KS test:",
            )
            @test !contains(
                string(MCBench.Plots.current()[1][:title]),
                "Reference t-test",
            )

            iid_plot_without_reference_test = only(MCBench.plot_teststatistic(
                testcase,
                metric;
                show_reference_test=false,
                save_plots=false,
            ))
            @test !contains(
                string(iid_plot_without_reference_test[1][:title]),
                "Reference t-test",
            )

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

            # The automatically chosen limits must include the full error bar,
            # not just its central marker.
            overview_with_wide_error = MCBench.plot_metrics(
                testcase,
                [(name="wide error", val=4.0, std=2.0)];
                save_plots=false,
            )
            @test last(MCBench.Plots.xlims(overview_with_wide_error)) > 6.0

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
            @test count(
                series -> series[:seriestype] == :scatter,
                quantile_reference_overview.series_list,
            ) == 1

            quantile_histograms = MCBench.plot_teststatistic(
                testcase,
                quantile_metric;
                save_plots=false,
            )
            @test length(quantile_histograms[1].series_list) == 2
            @test length(quantile_histograms[2].series_list) == 1
            @test length(quantile_histograms[3].series_list) == 1

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
            expected_sem = std([-0.1, 0.0, 0.1, 0.2, 0.3]) / sqrt(5)
            expected_normalized_difference = (0.1 - 0.05) / expected_sem
            @test only(marker_series[:x]) ≈ expected_normalized_difference
            @test only(marker_series[:xerror]) == 1.0
            labels = string.(getindex.(unsaved_reference_overview.series_list, :label))
            @test "Difference ± 1 SEM" in labels
            @test all(label -> !(label in labels), ["1σ region", "2σ region", "3σ region"])
            reference_yticks = unsaved_reference_overview[1][:yaxis][:ticks][2]
            @test only(reference_yticks) ==
                "Mean(x) (p = $(round(expected_reference_test.pvalue; sigdigits=4)))"

            missing_reference = standard_normal_testcase(1; info="Missing-Reference")
            @test_throws ArgumentError MCBench.plot_reference_metrics(
                missing_reference,
                [metric],
                sampler;
            )

            zero_sem_testcase = MCBench.Testcases(
                Normal(),
                bounds,
                1,
                "Zero-SEM-Test";
                reference_values=(marginal_mean=0.0,),
            )
            write_lines(
                joinpath(
                    "teststatistics_sampler",
                    "Zero-SEM-Test-Mean-Compared.txt",
                ),
                ["[0.1]", "[0.1]", "[0.1]"],
            )
            @test_throws ArgumentError MCBench.plot_reference_metrics(
                zero_sem_testcase,
                [metric],
                sampler;
                save_plots=false,
            )

            MCBench.Plots.closeall()
        end
    end
end
