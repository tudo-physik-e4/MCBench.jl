@testset "Plot generation" begin
    mktempdir() do dir
        cd(dir) do
            mkpath("teststatistics")
            mkpath("teststatistics_sampler")
            testcase = standard_normal_testcase(1; info="Plot-Test")
            metric = MCBench.marginal_mean()
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

            MCBench.plot_teststatistic(testcase, metric)
            iid_plot = joinpath("teststatistics", "Plot-Test-Mean-x1.pdf")
            @test isfile(iid_plot)
            @test filesize(iid_plot) > 0

            MCBench.plot_teststatistic(testcase, metric, sampler; nbins=3)
            comparison_plot = joinpath("Plot-Test", "Plot-Test-MeanCompared-x1.pdf")
            @test isfile(comparison_plot)
            @test filesize(comparison_plot) > 0

            MCBench.plot_metrics(testcase, [metric], sampler; names=["x"])
            metrics_pdf = joinpath("Plot-Test", "Plot-Test-Compared-metrics.pdf")
            metrics_png = joinpath("Plot-Test", "Plot-Test-Compared-metrics.png")
            @test isfile(metrics_pdf)
            @test isfile(metrics_png)
            @test filesize(metrics_pdf) > 0
            @test filesize(metrics_png) > 0

            MCBench.Plots.closeall()
        end
    end
end
