module SolverLogging_v1

using Logging
using Formatting

export
    SolverLogger,
    InnerLoop,
    OuterLoop,
    print_row,
    print_level,
    print_header,
    add_level!

include("logger.jl")

end # module
