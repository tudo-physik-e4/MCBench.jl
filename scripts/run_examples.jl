include(joinpath(@__DIR__, "..", "examples", "basic_metrics.jl"))
include(joinpath(@__DIR__, "..", "examples", "end_to_end_benchmark.jl"))

"""Run both maintained examples and write plots below `examples/output`."""
function main()
    BasicMetricsExample.main()
    EndToEndBenchmarkExample.main(
        output_dir=joinpath(@__DIR__, "..", "examples", "output"),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
