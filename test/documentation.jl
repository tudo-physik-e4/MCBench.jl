@testset "Documented example API" begin
    example_api = (
        :AnalyticReferenceDistribution,
        :BATMH,
        :CsvBasedSampler,
        :DsvSampler,
        :DsvTestcase,
        :Testcases,
        :TestMetric,
        :build_teststatistic,
        :get_effective_sample_size,
        :make_dsv,
        :marginal_mean,
        :marginal_quantiles,
        :marginal_skewness,
        :marginal_variance,
        :maximum_mean_discrepancy,
        :normal_3d_uncorrelated,
        :nonlinear_5d_mixture_laplace_t_easy,
        :nonlinear_5d_mixture_laplace_t_easy_params,
        :nonlinear_5d_mixture_laplace_t_hard,
        :nonlinear_5d_mixture_laplace_t_hard_params,
        :plot_metrics,
        :plot_reference_distribution,
        :plot_reference_metrics,
        :plot_teststatistic,
        :print_metric_summary,
        :read_effective_sample_sizes,
        :read_matched_iid_sample_size,
        :read_teststatistic,
        :reference_distribution_test,
        :reference_values,
        :sample,
        :set_mask,
        :sliced_wasserstein_distance,
    )

    for name in example_api
        binding = Base.Docs.Binding(MCBench, name)
        @test !isnothing(Base.Docs.doc(binding))
    end
end
