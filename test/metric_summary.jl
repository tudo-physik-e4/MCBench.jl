@testset "Metric text summaries" begin
    mktempdir() do dir
        cd(dir) do
            mkpath("teststatistics")
            mkpath("teststatistics_sampler")

            testcase = MCBench.Testcases(
                Normal(),
                NamedTupleDist(x=[-5..5]),
                1,
                "Summary-Test";
                reference_values=(
                    marginal_mean=0.05,
                    marginal_quantiles=metric -> Union{Missing,Float64}[
                        0.0,
                        missing,
                        1.0,
                    ],
                ),
            )
            metric = MCBench.marginal_mean()
            sampler = MCBench.DsvSampler(
                [MCBench.make_dsv(reshape(collect(-2.0:2.0), 1, :))];
                info="Candidate",
            )

            write_lines(
                joinpath("teststatistics", "Summary-Test-Mean.txt"),
                ["[-0.2]", "[-0.1]", "[0.0]", "[0.1]", "[0.2]"],
            )
            write_lines(
                joinpath("teststatistics_sampler", "Summary-Test-Mean-Candidate.txt"),
                ["[-0.1]", "[0.0]", "[0.1]", "[0.2]", "[0.3]"],
            )
            write_lines(
                joinpath("teststatistics", "Summary-Test-Quantiles-50-90-99.txt"),
                ["[0.0, 0.9, 0.99]", "[0.1, 1.0, 1.1]", "[-0.1, 0.8, 0.9]"],
            )
            write_lines(
                joinpath(
                    "teststatistics_sampler",
                    "Summary-Test-Quantiles-50-90-99-Candidate.txt",
                ),
                ["[0.1, 1.0, 1.1]", "[0.2, 1.1, 1.2]", "[0.0, 0.9, 1.0]"],
            )

            iid_rows = MCBench.metric_summary_rows(
                testcase,
                [metric],
                sampler;
                names=["x"],
            )
            @test length(iid_rows) == 1
            @test iid_rows[1].metric == "Mean(x)"
            @test iid_rows[1].sampler_mean ≈ 0.1
            @test iid_rows[1].sampler_std ≈ std([-0.1, 0.0, 0.1, 0.2, 0.3])
            @test iid_rows[1].sampler_sem ≈
                std([-0.1, 0.0, 0.1, 0.2, 0.3]) / sqrt(5)
            @test abs(iid_rows[1].comparison_mean) < eps(Float64)
            @test iid_rows[1].comparison_std ≈ std([-0.2, -0.1, 0.0, 0.1, 0.2])
            @test isnothing(iid_rows[1].reference_value)
            @test iid_rows[1].difference ≈ 0.1
            @test iid_rows[1].standardized_difference ≈
                0.1 / std([-0.2, -0.1, 0.0, 0.1, 0.2])
            expected_ks = MCBench.ks_test(
                [-0.2, -0.1, 0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1, 0.2, 0.3],
            )
            @test iid_rows[1].ks_statistic ≈ expected_ks.statistic
            @test iid_rows[1].ks_pvalue ≈ expected_ks.pvalue
            @test isnothing(iid_rows[1].reference_statistic)
            @test isnothing(iid_rows[1].reference_pvalue)

            iid_summary = MCBench.metric_summary(testcase, [metric], sampler; digits=4)
            @test contains(iid_summary, "IID comparison")
            @test contains(iid_summary, "Sampler mean")
            @test contains(iid_summary, "IID mean")
            @test contains(iid_summary, "KS D")
            @test contains(iid_summary, "KS p-value")
            @test contains(iid_summary, "Mean(x1)")

            iid_summary_with_reference = MCBench.metric_summary(
                testcase,
                [metric],
                sampler;
                include_reference=true,
                digits=4,
            )
            @test contains(iid_summary_with_reference, "Reference")
            @test contains(iid_summary_with_reference, "Reference p-value")
            @test contains(iid_summary_with_reference, "0.05")
            referenced_iid_rows = MCBench.metric_summary_rows(
                testcase,
                [metric],
                sampler;
                include_reference=true,
            )
            @test referenced_iid_rows[1].reference_value == 0.05
            expected_reference_test = MCBench.reference_value_test(
                [-0.1, 0.0, 0.1, 0.2, 0.3],
                0.05,
            )
            @test referenced_iid_rows[1].reference_statistic ≈
                expected_reference_test.statistic
            @test referenced_iid_rows[1].reference_pvalue ≈
                expected_reference_test.pvalue

            quantile_metric = MCBench.marginal_quantiles()
            partial_reference_rows = MCBench.metric_summary_rows(
                testcase,
                [quantile_metric],
                sampler;
                include_reference=true,
            )
            @test isequal(
                getproperty.(partial_reference_rows, :reference_value),
                [0.0, missing, 1.0],
            )
            @test isnothing(partial_reference_rows[2].reference_pvalue)
            partial_reference_summary = MCBench.metric_summary(
                testcase,
                [quantile_metric],
                sampler;
                include_reference=true,
            )
            @test contains(partial_reference_summary, "—")

            output = IOBuffer()
            printed = MCBench.print_metric_summary(
                testcase,
                [metric],
                sampler;
                io=output,
                names=["parameter"],
            )
            @test String(take!(output)) == printed * "\n"
            @test contains(printed, "Mean(parameter)")

            reference_rows = MCBench.metric_summary_rows(
                testcase,
                [MCBench.marginal_variance(), metric],
                sampler;
                comparison=:reference,
            )
            @test length(reference_rows) == 1
            @test reference_rows[1].comparison_mean == 0.05
            @test reference_rows[1].reference_value == 0.05
            @test isnothing(reference_rows[1].comparison_std)
            @test reference_rows[1].difference ≈ 0.05
            @test reference_rows[1].standardized_difference ≈
                0.05 / (std([-0.1, 0.0, 0.1, 0.2, 0.3]) / sqrt(5))
            @test isnothing(reference_rows[1].ks_statistic)
            @test isnothing(reference_rows[1].ks_pvalue)
            @test reference_rows[1].reference_statistic ≈
                expected_reference_test.statistic
            @test reference_rows[1].reference_pvalue ≈
                expected_reference_test.pvalue

            quantile_reference_rows = MCBench.metric_summary_rows(
                testcase,
                [quantile_metric],
                sampler;
                comparison=:reference,
            )
            @test length(quantile_reference_rows) == 2
            @test getproperty.(quantile_reference_rows, :metric) == [
                "50% quantile (x1)",
                "99% quantile (x1)",
            ]

            reference_summary = MCBench.metric_summary(
                testcase,
                [metric],
                sampler;
                comparison=:reference,
            )
            @test contains(reference_summary, "REFERENCE comparison")
            @test contains(reference_summary, "Reference")
            @test contains(reference_summary, "Sampler SEM")
            @test contains(reference_summary, "Δ / SEM")
            @test contains(reference_summary, "Reference p-value")

            no_references = standard_normal_testcase(1; info="No-References")
            @test_throws ArgumentError MCBench.metric_summary(
                no_references,
                [metric],
                sampler;
                comparison=:reference,
            )
            @test_throws ArgumentError MCBench.metric_summary(
                testcase,
                [metric],
                sampler;
                comparison=:unknown,
            )
            @test_throws ArgumentError MCBench.metric_summary(
                testcase,
                [metric],
                sampler;
                digits=0,
            )
            @test_throws DimensionMismatch MCBench.metric_summary(
                testcase,
                [metric],
                sampler;
                names=["x", "y"],
            )
        end
    end
end
