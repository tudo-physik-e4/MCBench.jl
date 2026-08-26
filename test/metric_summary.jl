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
                reference_values=(marginal_mean=0.05,),
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
            @test abs(iid_rows[1].comparison_mean) < eps(Float64)
            @test iid_rows[1].comparison_std ≈ std([-0.2, -0.1, 0.0, 0.1, 0.2])
            @test iid_rows[1].difference ≈ 0.1
            @test iid_rows[1].standardized_difference ≈
                0.1 / std([-0.2, -0.1, 0.0, 0.1, 0.2])
            expected_ks = MCBench.ks_test(
                [-0.2, -0.1, 0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1, 0.2, 0.3],
            )
            @test iid_rows[1].ks_statistic ≈ expected_ks.statistic
            @test iid_rows[1].ks_pvalue ≈ expected_ks.pvalue

            iid_summary = MCBench.metric_summary(testcase, [metric], sampler; digits=4)
            @test contains(iid_summary, "IID comparison")
            @test contains(iid_summary, "Sampler mean")
            @test contains(iid_summary, "IID mean")
            @test contains(iid_summary, "KS D")
            @test contains(iid_summary, "KS p-value")
            @test contains(iid_summary, "Mean(x1)")

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
            @test isnothing(reference_rows[1].comparison_std)
            @test reference_rows[1].difference ≈ 0.05
            @test reference_rows[1].standardized_difference ≈
                0.05 / std([-0.1, 0.0, 0.1, 0.2, 0.3])
            @test isnothing(reference_rows[1].ks_statistic)
            @test isnothing(reference_rows[1].ks_pvalue)

            reference_summary = MCBench.metric_summary(
                testcase,
                [metric],
                sampler;
                comparison=:reference,
            )
            @test contains(reference_summary, "REFERENCE comparison")
            @test contains(reference_summary, "Reference")
            @test contains(reference_summary, "Δ / σ_sampler")

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
