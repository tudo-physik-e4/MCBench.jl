@testset "Built-in example distributions" begin
    @testset "Gaussian analytic reference distributions" begin
        for (testcase, dimension) in (
            (MCBench.normal_1d_uncorrelated, 1),
            (MCBench.normal_3d_uncorrelated, 3),
            (MCBench.normal_10d_strongly_correlated, 10),
        )
            reference = MCBench.reference_distribution(
                testcase,
                :squared_mahalanobis,
            )
            @test reference isa MCBench.AnalyticReferenceDistribution
            @test reference.distribution == Chisq(dimension)
        end
        @test isnothing(MCBench.reference_distribution(
            MCBench.cauchy_1d,
            :squared_mahalanobis,
        ))
    end

    @testset "Nonlinear 5D Mixture-Laplace-t" begin
        easy = MCBench.nonlinear_5d_mixture_laplace_t_easy
        hard = MCBench.nonlinear_5d_mixture_laplace_t_hard

        @test easy.dim == 5
        @test hard.dim == 5
        @test easy.info == "Nonlinear-5D-Mixture-Laplace-t-Easy"
        @test hard.info == "Nonlinear-5D-Mixture-Laplace-t-Hard"
        @test length(easy.f) == 5

        easy_params = MCBench.nonlinear_5d_mixture_laplace_t_easy_params
        hard_params = MCBench.nonlinear_5d_mixture_laplace_t_hard_params
        @test easy_params.A == 2.0
        @test easy_params.ω == 0.2
        @test hard_params.mode_sep1 == 8.0
        @test hard_params.mix_p1 == 0.3
        @test hard_params.ω == 0.7

        modified = MCBench.withparams(easy_params; A=4.0, κ=2.0)
        @test modified.A == 4.0
        @test modified.κ == 2.0
        @test easy_params.A == 2.0
        @test_throws ArgumentError MCBench.NonlinearMixtureLaplaceTParams(σ1=0)
        @test_throws ArgumentError MCBench.NonlinearMixtureLaplaceTParams(mix_p1=1)
        @test_throws ArgumentError MCBench.NonlinearMixtureLaplaceTParams(ν3=2)
        @test_throws ArgumentError MCBench.NonlinearMixtureLaplaceTParams(η4=-0.1)

        custom = MCBench.nonlinear_5d_mixture_laplace_t(modified; info="Custom")
        @test custom.f.params == modified
        @test custom.info == "Custom"

        first_samples = rand(MersenneTwister(42), easy.f, 20)
        repeated_samples = rand(MersenneTwister(42), easy.f, 20)
        @test first_samples == repeated_samples
        @test size(first_samples) == (5, 20)
        @test all(isfinite, first_samples)

        log_densities = logpdf(easy.f, first_samples)
        @test length(log_densities) == 20
        @test all(isfinite, log_densities)
        @test log_densities[1] ≈ logpdf(easy.f, first_samples[:, 1])
        @test_throws DimensionMismatch logpdf(easy.f, zeros(4))
        @test_throws DimensionMismatch logpdf(easy.f, zeros(4, 2))

        easy_mean = MCBench.reference_values(easy, MCBench.marginal_mean())
        hard_mean = MCBench.reference_values(hard, MCBench.marginal_mean())
        easy_variance = MCBench.reference_values(easy, MCBench.marginal_variance())
        easy_quantiles = MCBench.reference_values(easy, MCBench.marginal_quantiles())
        hard_quantiles = MCBench.reference_values(hard, MCBench.marginal_quantiles())
        @test easy_mean ≈ zeros(5) atol=1e-14
        @test hard_mean[1] ≈ -3.2
        @test all(>(0), easy_variance)
        @test easy_quantiles[1:5] ≈ zeros(5) atol=1e-14
        @test easy_quantiles[6] ≈ quantile(Normal(0, easy_params.σ1), 0.9)
        @test all(ismissing, easy_quantiles[7:10])
        @test easy_quantiles[11] ≈ quantile(Normal(0, easy_params.σ1), 0.99)
        @test all(ismissing, easy_quantiles[12:15])
        @test isequal(hard_quantiles[1:5], [missing, missing, missing, 0.0, 0.0])
        @test all(ismissing, hard_quantiles[6:15])
        @test isnothing(MCBench.reference_values(
            easy,
            MCBench.sliced_wasserstein_distance(),
        ))
        @test isnothing(MCBench.reference_values(
            easy,
            MCBench.maximum_mean_discrepancy(),
        ))

        testcase_samples = MCBench.sample(easy, 10)
        @test length(testcase_samples) == 10
        @test all(isfinite, testcase_samples.logd)
    end

    @testset "PosteriorDB eight schools" begin
        centered = MCBench.eight_schools_testcase
        transformed = MCBench.eight_schools_testcase_trafo

        @test centered.dim == 10
        @test transformed.dim == 10
        @test centered.info == "EightSchoolsAcceptReject"
        @test transformed.info == "EightSchoolsAcceptReject-Transformed"
        @test !centered.f.transformed
        @test transformed.f.transformed

        point = [zeros(8); 0.0; 1.0]
        transformed_point = [zeros(8); 0.0; 0.0]
        @test isfinite(logpdf(centered.f, point))
        @test isfinite(logpdf(transformed.f, transformed_point))
        @test logpdf(centered.f, point) ≈ logpdf(transformed.f, transformed_point)
        @test logpdf(centered.f, reshape(point, 10, 1)) == [
            logpdf(centered.f, point),
        ]
        @test logpdf(centered.f, [zeros(8); 0.0; -1.0]) == -Inf
        @test_throws DimensionMismatch logpdf(centered.f, zeros(9))
        @test_throws DimensionMismatch logpdf(centered.f, zeros(9, 2))

        first_sample = rand(MersenneTwister(7), centered.f, 1)
        repeated_sample = rand(MersenneTwister(7), centered.f, 1)
        @test first_sample == repeated_sample
        @test size(first_sample) == (10, 1)
        @test first_sample[10, 1] > 0

        custom = MCBench.make_eight_schools_testcase(info="Custom-Eight-Schools")
        @test custom.info == "Custom-Eight-Schools"
        @test custom.f.data == centered.f.data
    end
end
