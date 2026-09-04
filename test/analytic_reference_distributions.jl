@testset "Analytic reference distributions" begin
    @testset "testcase metadata stays separate" begin
        analytic_reference = MCBench.AnalyticReferenceDistribution(
            x -> sum(abs2, x),
            Chisq(2);
            info="Squared radius",
        )
        testcase = MCBench.Testcases(
            MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)),
            NamedTupleDist(x=fill(-5..5, 2)),
            2,
            "Analytic-Metadata";
            reference_values=(marginal_mean=zeros(2),),
            reference_distributions=(squared_radius=analytic_reference,),
        )

        @test MCBench.reference_distribution(testcase, :squared_radius) ===
            analytic_reference
        @test MCBench.reference_distribution(testcase, "squared_radius") ===
            analytic_reference
        @test isnothing(MCBench.reference_distribution(testcase, :unknown))
        @test MCBench.reference_values(testcase, MCBench.marginal_mean()) == zeros(2)
        @test !hasproperty(testcase.reference_values, :squared_radius)

        @test_throws ArgumentError MCBench.Testcases(
            Normal(),
            1,
            "Invalid-Reference-Container";
            reference_distributions=[:not_a_named_tuple],
        )
        @test_throws ArgumentError MCBench.Testcases(
            Normal(),
            1,
            "Invalid-Reference-Entry";
            reference_distributions=(bad=Normal(),),
        )
    end

    @testset "Gaussian Mahalanobis observable and KS test" begin
        location = [1.0, -2.0]
        covariance = [2.0 0.5; 0.5 1.0]
        distribution = MvNormal(location, covariance)
        reference = MCBench.mahalanobis_reference(distribution)
        testcase = MCBench.Testcases(
            distribution,
            NamedTupleDist(x=fill(-10..10, 2)),
            2,
            "Mahalanobis-Test";
            reference_distributions=(squared_mahalanobis=reference,),
        )

        coordinates = [
            1.0 2.0 -1.0
            -2.0 -1.0 0.0
        ]
        result = MCBench.reference_distribution_test(
            testcase,
            coordinates,
            :squared_mahalanobis,
        )
        expected_values = [
            dot(column - location, covariance \ (column - location))
            for column in eachcol(coordinates)
        ]
        @test result.name == :squared_mahalanobis
        @test result.observable_values ≈ expected_values
        @test result.reference.distribution == Chisq(2)
        expected_test = MCBench.ApproximateOneSampleKSTest(
            expected_values,
            Chisq(2),
        )
        @test result.statistic ≈ expected_test.δ
        @test result.pvalue ≈ MCBench.pvalue(expected_test; tail=:both)
        @test result.effective_sample_size == length(expected_values)
        @test contains(sprint(show, result), "Squared Mahalanobis distance")

        Random.seed!(20260831)
        target_samples = rand(distribution, 5_000)
        calibrated = MCBench.reference_distribution_test(
            testcase,
            target_samples,
            :squared_mahalanobis,
        )
        @test calibrated.statistic < 0.03
        @test 0 <= calibrated.pvalue <= 1

        shifted_samples = target_samples .+ [2.0, 2.0]
        shifted = MCBench.reference_distribution_test(
            testcase,
            shifted_samples,
            :squared_mahalanobis,
        )
        @test shifted.statistic > calibrated.statistic
        @test shifted.pvalue < 1e-10

        univariate_reference = MCBench.mahalanobis_reference(Normal(2, 3))
        @test univariate_reference.observable([5.0]) ≈ 1.0
        @test univariate_reference.distribution == Chisq(1)
    end

    @testset "ESS-adjusted KS calibration" begin
        values = collect(range(-2.5, 2.5; length=200))
        raw_test = MCBench.ApproximateOneSampleKSTest(values, Normal())
        raw_result = MCBench.one_sample_ks_test(values, Normal())
        adjusted_result = MCBench.one_sample_ks_test(
            values,
            Normal();
            effective_sample_size=40.9,
        )
        expected_adjusted_test = MCBench.ApproximateOneSampleKSTest(
            40,
            raw_test.δ,
            raw_test.δp,
            raw_test.δn,
        )

        @test adjusted_result.statistic == raw_result.statistic
        @test adjusted_result.pvalue ≈ MCBench.pvalue(
            expected_adjusted_test;
            tail=:both,
        )
        @test adjusted_result.pvalue > raw_result.pvalue

        reference = MCBench.AnalyticReferenceDistribution(x -> x[1], Normal())
        testcase = MCBench.Testcases(
            Normal(),
            1,
            "ESS-Adjusted-Analytic";
            reference_distributions=(coordinate=reference,),
        )
        result = MCBench.reference_distribution_test(
            testcase,
            values,
            :coordinate;
            effective_sample_size=40.9,
        )
        @test result.effective_sample_size == 40
        @test result.statistic == raw_result.statistic
        @test result.pvalue == adjusted_result.pvalue
        @test contains(sprint(show, result), "n_eff=40")

        sampler = FixedEffectiveSampleSizeSampler(200, 40.9, 0, "Analytic-ESS")
        sampler_result = MCBench.reference_distribution_test(
            testcase,
            sampler,
            :coordinate,
        )
        @test sampler_result.effective_sample_size == 40
        @test length(sampler_result.observable_values) == 200

        uncorrected_sampler = FixedEffectiveSampleSizeSampler(
            200,
            40.9,
            0,
            "Analytic-Raw-N",
        )
        uncorrected_result = MCBench.reference_distribution_test(
            testcase,
            uncorrected_sampler,
            :coordinate;
            correct_for_ess=false,
        )
        @test uncorrected_result.effective_sample_size == 200

        @test_throws ArgumentError MCBench.one_sample_ks_test(
            values,
            Normal();
            effective_sample_size=0,
        )
        @test_throws ArgumentError MCBench.one_sample_ks_test(
            values,
            Normal();
            effective_sample_size=length(values) + 1,
        )
        @test_throws ArgumentError MCBench.reference_distribution_test(
            testcase,
            values,
            :coordinate;
            effective_sample_size=NaN,
        )
    end

    @testset "custom goodness-of-fit strategy" begin
        # This discrete example exercises the extension point intended for an
        # exact energy test without pretending its probabilities are IID samples.
        custom_test = (values, distribution) -> (
            statistic=abs(mean(values) - mean(distribution)),
            pvalue=0.75,
        )
        reference = MCBench.AnalyticReferenceDistribution(
            x -> x[1] >= 0 ? 1.0 : 0.0,
            DiscreteNonParametric([0.0, 1.0], [0.25, 0.75]);
            info="Binary energy",
            goodness_of_fit=custom_test,
            test_name="Custom discrete test",
        )
        testcase = MCBench.Testcases(
            Normal(),
            1,
            "Discrete-Analytic";
            reference_distributions=(energy=reference,),
        )
        result = MCBench.reference_distribution_test(
            testcase,
            [-2.0, 1.0, 2.0, 3.0],
            :energy,
        )
        @test result.observable_values == [0.0, 1.0, 1.0, 1.0]
        @test result.statistic == 0.0
        @test result.pvalue == 0.75
        @test_throws ArgumentError MCBench.one_sample_ks_test(
            result.observable_values,
            reference.distribution,
        )

        @test_throws ArgumentError MCBench.reference_distribution_test(
            testcase,
            [-2.0, 1.0, 2.0, 3.0],
            :energy;
            effective_sample_size=2,
        )
    end

    @testset "input validation and weighted samples" begin
        good_reference = MCBench.AnalyticReferenceDistribution(
            x -> x[1],
            Normal(),
        )
        testcase = MCBench.Testcases(
            Normal(),
            1,
            "Analytic-Validation";
            reference_distributions=(coordinate=good_reference,),
        )
        samples = MCBench.make_dsv(
            reshape([-1.0, 0.0, 1.0], 1, :);
            weights=[1.0, 1.0, 4.0],
        )
        @test_throws ArgumentError MCBench.reference_distribution_test(
            testcase,
            samples,
            :coordinate;
            unweight=false,
        )
        Random.seed!(101)
        unweighted = MCBench.reference_distribution_test(
            testcase,
            samples,
            :coordinate,
        )
        @test !isempty(unweighted.observable_values)

        @test_throws ArgumentError MCBench.reference_distribution_test(
            testcase,
            Float64[],
            :coordinate,
        )
        @test_throws ArgumentError MCBench.reference_distribution_test(
            testcase,
            [1.0, 2.0],
            :missing,
        )
        @test_throws DimensionMismatch MCBench.reference_distribution_test(
            MCBench.normal_2d_uncorrelated,
            [1.0, 2.0],
            :squared_mahalanobis,
        )

        nonfinite_reference = MCBench.AnalyticReferenceDistribution(
            x -> NaN,
            Normal(),
        )
        nonfinite_testcase = MCBench.Testcases(
            Normal(),
            1,
            "Nonfinite-Observable";
            reference_distributions=(bad=nonfinite_reference,),
        )
        @test_throws ArgumentError MCBench.reference_distribution_test(
            nonfinite_testcase,
            [0.0, 1.0],
            :bad,
        )

        invalid_test_reference = MCBench.AnalyticReferenceDistribution(
            x -> x[1],
            Normal();
            goodness_of_fit=(values, distribution) -> (statistic=0.1, pvalue=2.0),
        )
        invalid_testcase = MCBench.Testcases(
            Normal(),
            1,
            "Invalid-GOF";
            reference_distributions=(bad=invalid_test_reference,),
        )
        @test_throws ArgumentError MCBench.reference_distribution_test(
            invalid_testcase,
            [0.0, 1.0],
            :bad,
        )
    end

    @testset "plots use the analytic law directly" begin
        mktempdir() do dir
            cd(dir) do
                testcase = MCBench.normal_2d_uncorrelated
                Random.seed!(91)
                samples = MCBench.sample(testcase, 500)
                result = MCBench.reference_distribution_test(
                    testcase,
                    samples,
                    :squared_mahalanobis,
                )

                unsaved = MCBench.plot_reference_distribution(
                    result;
                    nbins=20,
                    save_plots=false,
                )
                @test unsaved isa MCBench.Plots.Plot
                @test length(unsaved.subplots) == 2
                @test contains(string(unsaved[1][:title]), "One-sample KS")
                @test contains(string(unsaved[1][:title]), "D =")
                @test contains(string(unsaved[1][:title]), "p =")
                @test !contains(string(unsaved[1][:title]), testcase.info)

                customized = MCBench.plot_reference_distribution(
                    result;
                    nbins=20,
                    title="Custom reference title",
                    distribution_legend=:bottomleft,
                    cdf_legend=:bottomright,
                    panel_heights=(3, 1),
                    save_plots=false,
                )
                @test string(customized[1][:title]) == "Custom reference title"
                @test customized[1][:legend_position] == :bottomleft
                @test customized[2][:legend_position] == :bottomright
                @test MCBench._normalized_panel_heights((3, 1)) == (0.75, 0.25)
                @test_throws ArgumentError MCBench.plot_reference_distribution(
                    result;
                    panel_heights=3,
                    save_plots=false,
                )
                @test_throws ArgumentError MCBench.plot_reference_distribution(
                    result;
                    panel_heights=(1, 0),
                    save_plots=false,
                )

                adjusted = MCBench.reference_distribution_test(
                    testcase,
                    samples,
                    :squared_mahalanobis;
                    effective_sample_size=100,
                )
                adjusted_plot = MCBench.plot_reference_distribution(
                    adjusted;
                    nbins=20,
                    save_plots=false,
                )
                @test contains(string(adjusted_plot[1][:title]), "ESS-adjusted")
                @test contains(
                    string(adjusted_plot[1][:title]),
                    "\nESS-adjusted: n_eff = 100",
                )
                empirical_label = string(
                    adjusted_plot.series_list[1][:label],
                )
                @test contains(empirical_label, "ESS = 100")

                output_path = MCBench.plot_reference_distribution(
                    testcase,
                    samples,
                    :squared_mahalanobis;
                    nbins=20,
                )
                expected_path = joinpath(
                    testcase.info,
                    "$(testcase.info)-squared_mahalanobis-analytic-reference.pdf",
                )
                @test output_path == expected_path
                @test isfile(output_path)
                @test filesize(output_path) > 0
            end
        end
    end
end
