@testset "Metric types and calculations" begin
    testcase = standard_normal_testcase(2; info="Metric-Test")
    values = [-2.0 -1.0 0.0 1.0 2.0; 2.0 1.0 0.0 1.0 2.0]
    logdensities = [-4.0, -1.0, 0.0, -1.0, -4.0]
    samples = MCBench.make_dsv(values, logdensities)

    @testset "constructors" begin
        @test MCBench.marginal_mean().info == "Mean"
        @test MCBench.marginal_variance().info == "Variance"
        @test MCBench.marginal_quantiles().probabilities == (0.5, 0.9, 0.99)
        @test MCBench.marginal_quantiles().info == "Quantiles-50-90-99"
        @test MCBench.marginal_quantile().probabilities == (0.5,)
        @test MCBench.marginal_quantile(percent=95).probabilities == (0.95,)
        @test MCBench.marginal_quantiles([0.25, 0.75]).probabilities == (0.25, 0.75)
        @test MCBench.marginal_quantiles(percentages=[10, 99]).probabilities == (0.1, 0.99)
        @test_throws ArgumentError MCBench.marginal_quantiles(1.1)
        @test_throws ArgumentError MCBench.marginal_quantiles(percent=101)
        @test_throws ArgumentError MCBench.marginal_quantiles(Float64[])
        @test_throws ArgumentError MCBench.marginal_quantiles([0.5, 0.5])
        @test MCBench.global_mode().info == "Globalmode"
        @test MCBench.marginal_mode().info == "Marginalmode"
        @test MCBench.marginal_skewness().info == "Skewness"
        @test MCBench.marginal_kurtosis().info == "Kurtosis"
        @test MCBench.wasserstein_1d().info == "Wasserstein"
        @test MCBench.sliced_wasserstein_distance(0.0, 12).N == 12
        @test MCBench.maximum_mean_discrepancy(0.0, 13).N == 13
        @test MCBench.chi_squared().info == "Chi-Squared"
    end

    @testset "one-sample metrics" begin
        means = MCBench.calc_metric(testcase, samples, MCBench.marginal_mean())
        variances = MCBench.calc_metric(testcase, samples, MCBench.marginal_variance())
        quantile_metric = MCBench.marginal_quantiles([0.25, 0.5, 0.75])
        quantiles = MCBench.calc_metric(testcase, samples, quantile_metric)
        global_modes = MCBench.calc_metric(testcase, samples, MCBench.global_mode())
        marginal_modes = MCBench.calc_metric(testcase, samples, MCBench.marginal_mode())
        skewnesses = MCBench.calc_metric(testcase, samples, MCBench.marginal_skewness())
        kurtoses = MCBench.calc_metric(testcase, samples, MCBench.marginal_kurtosis())

        @test [metric.val for metric in means] ≈ [0.0, 1.2]
        @test [metric.val for metric in variances] ≈ [2.5, 0.7]
        weights = FrequencyWeights(ones(size(values, 2)))
        expected_quantiles = [
            StatsBase.quantile(row, weights, probability)
            for probability in quantile_metric.probabilities
            for row in eachrow(values)
        ]
        @test [metric.val for metric in quantiles] ≈ expected_quantiles
        @test MCBench.metric_output_dimension(testcase, quantile_metric) == 6
        @test MCBench.metric_output_labels(
            testcase,
            quantile_metric;
            names=["a", "b"],
        ) == [
            "25% quantile (a)",
            "25% quantile (b)",
            "50% quantile (a)",
            "50% quantile (b)",
            "75% quantile (a)",
            "75% quantile (b)",
        ]

        weighted_samples = MCBench.make_dsv(
            values,
            logdensities;
            weights=[1.0, 1.0, 1.0, 1.0, 5.0],
        )
        weighted_median = MCBench.calc_metric(
            testcase,
            weighted_samples,
            MCBench.marginal_quantile(),
        )
        expected_weighted_median = [
            StatsBase.quantile(row, FrequencyWeights(weighted_samples.weight), 0.5)
            for row in eachrow(values)
        ]
        @test [metric.val for metric in weighted_median] ≈ expected_weighted_median

        @test [metric.val for metric in global_modes] == [0.0, 0.0]
        @test length(marginal_modes) == 2
        @test all(isfinite(metric.val) for metric in marginal_modes)

        expected_skewness = [StatsBase.skewness(row, weights) for row in eachrow(values)]
        expected_kurtosis = [StatsBase.kurtosis(row, weights) for row in eachrow(values)]
        @test [metric.val for metric in skewnesses] ≈ expected_skewness
        @test [metric.val for metric in kurtoses] ≈ expected_kurtosis

        run_result = MCBench.run_teststatistic(testcase, samples, MCBench.marginal_mean(), 1)
        @test [metric.val for metric in run_result] ≈ [0.0, 1.2]

        sampler_result = MCBench.run_teststatistic(
            testcase,
            samples,
            MCBench.marginal_mean(),
            MCBench.IIDSampler(),
        )
        @test [metric.val for metric in sampler_result] ≈ [0.0, 1.2]
    end
end
