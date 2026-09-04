@testset "Runnable examples" begin
    runner_path = joinpath(@__DIR__, "..", "scripts", "run_examples.jl")
    @test Meta.parseall(read(runner_path, String)) isa Expr
    quickstart_path = joinpath(@__DIR__, "..", "examples", "quickstart.jl")
    @test Meta.parseall(read(quickstart_path, String)) isa Expr
    paper_example_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "paper_section_6_1.jl",
    )
    paper_example = read(paper_example_path, String)
    @test Meta.parseall(paper_example) isa Expr
    @test contains(paper_example, "marginal_quantiles()")
    @test contains(paper_example, "plot_reference_metrics")
    @test contains(paper_example, "include_reference=true")
    @test contains(paper_example, "comparison=:reference")
    @test contains(paper_example, "show_reference_test=true")
    @test contains(paper_example, "show_ks_test=true")
    @test contains(paper_example, "reference_distribution_test")
    @test contains(paper_example, "plot_reference_distribution")
    @test contains(paper_example, "x2_standardized_residual")
    @test contains(paper_example, "Laplace(0.0, 1.0)")
    @test contains(paper_example, ":x2_residual")
    @test contains(paper_example, "analytic_effective_sample_size")
    @test contains(
        paper_example,
        "effective_sample_size=analytic_effective_sample_size",
    )
    @test !contains(paper_example, "gaussian_testcase")
    @test contains(paper_example, "raw_paper_batches")
    @test contains(paper_example, "n_repetitions = paper_run ? 50 : 3")
    @test !contains(paper_example, "show_ks_test=false")
    @test !contains(paper_example, "paper_metrics")

    hard_paper_example_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "paper_section_6_2.jl",
    )
    hard_paper_example = read(hard_paper_example_path, String)
    @test Meta.parseall(hard_paper_example) isa Expr
    @test contains(
        hard_paper_example,
        "nonlinear_5d_mixture_laplace_t_hard",
    )
    @test contains(
        hard_paper_example,
        "nonlinear_5d_mixture_laplace_t_hard_params",
    )
    @test !contains(
        hard_paper_example,
        "nonlinear_5d_mixture_laplace_t_easy",
    )
    @test contains(hard_paper_example, "figure-2b-x1-marginal")
    @test contains(hard_paper_example, "figure-5-metric-overview")
    @test contains(hard_paper_example, "marginal_quantiles()")
    @test contains(hard_paper_example, "plot_reference_metrics")
    @test contains(hard_paper_example, "print_metric_summary")
    @test contains(hard_paper_example, "reference_distribution_test")
    @test contains(hard_paper_example, "plot_reference_distribution")
    @test contains(hard_paper_example, "x2_standardized_residual")
    @test contains(hard_paper_example, "Laplace(0.0, 1.0)")
    @test contains(hard_paper_example, "analytic_effective_sample_size")
    @test contains(hard_paper_example, "BAT.AssumeConvergence()")
    @test contains(hard_paper_example, "raw_paper_batches")
    @test contains(hard_paper_example, "n_repetitions = paper_run ? 50 : 3")
    @test !contains(hard_paper_example, "show_ks_test=false")

    analytic_example_path = joinpath(
        @__DIR__,
        "..",
        "examples",
        "analytic_reference_distribution.jl",
    )
    analytic_example = read(analytic_example_path, String)
    @test Meta.parseall(analytic_example) isa Expr
    @test contains(analytic_example, "reference_distribution_test")
    @test contains(analytic_example, "plot_reference_distribution")
    @test contains(analytic_example, "AnalyticReferenceDistribution")
    @test contains(analytic_example, "coordinate_sum")
    @test contains(analytic_example, "coordinate_sum_distribution")
    @test !contains(analytic_example, "MCBench.build_teststatistic(")

    include(joinpath(@__DIR__, "..", "examples", "basic_metrics.jl"))
    basic_result = BasicMetricsExample.main(n_samples=200, seed=77, verbose=false)
    @test length(basic_result.mean) == 2
    @test length(basic_result.variance) == 2
    @test isfinite(basic_result.wasserstein)
    @test isfinite(basic_result.mmd)
    @test basic_result.reference_mean == [0.0, 0.0]

    include(joinpath(@__DIR__, "..", "examples", "end_to_end_benchmark.jl"))
    mktempdir() do dir
        result = EndToEndBenchmarkExample.main(
            output_dir=dir,
            n_repetitions=3,
            n_samples=30,
            seed=88,
            verbose=false,
        )

        @test size(result.iid_mean) == (2, 3)
        @test size(result.sampler_mean) == (2, 3)
        @test length(result.plot_files) == 6
        @test any(contains("Variance"), result.plot_files)
        @test all(isfile, result.plot_files)
        @test all(path -> filesize(path) > 0, result.plot_files)
    end
end
