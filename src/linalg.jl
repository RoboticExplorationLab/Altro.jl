const USE_OCTAVIAN = haskey(ENV, "ALTRO_USE_OCTAVIAN")
import Octavian

use_octavian(flag::Bool=true) = global USE_OCTAVIAN = flag  # note this overwrites a constant

# function mul!(C,A,B)
#     if USE_OCTAVIAN
#         Octavian.matmul!(C,A,B)
#     else
#         mul!(C,A,B)
#     end
# end

# function mul!(C,A,B, α,β)
#     if USE_OCTAVIAN
#         Octavian.matmul!(C,A,B, α,β)
#     else
#         mul!(C,A,B, α,β)
#     end
# end