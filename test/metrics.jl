@testset "Metric types and calculations" begin
    testcase = standard_normal_testcase(2; info="Metric-Test")
    values = [-2.0 -1.0 0.0 1.0 2.0; 2.0 1.0 0.0 1.0 2.0]
    logdensities = [-4.0, -1.0, 0.0, -1.0, -4.0]
    samples = MCBench.make_dsv(values, logdensities)

    @testset "constructors" begin
        @test MCBench.marginal_mean().info == "Mean"
        @test MCBench.marginal_variance().info == "Variance"
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
        global_modes = MCBench.calc_metric(testcase, samples, MCBench.global_mode())
        marginal_modes = MCBench.calc_metric(testcase, samples, MCBench.marginal_mode())
        skewnesses = MCBench.calc_metric(testcase, samples, MCBench.marginal_skewness())
        kurtoses = MCBench.calc_metric(testcase, samples, MCBench.marginal_kurtosis())

        @test [metric.val for metric in means] ≈ [0.0, 1.2]
        @test [metric.val for metric in variances] ≈ [2.5, 0.7]
        @test [metric.val for metric in global_modes] == [0.0, 0.0]
        @test length(marginal_modes) == 2
        @test all(isfinite(metric.val) for metric in marginal_modes)

        weights = FrequencyWeights(ones(size(values, 2)))
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
