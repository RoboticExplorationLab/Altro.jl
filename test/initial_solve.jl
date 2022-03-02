ENV["ALTRO_USE_OCTAVIAN"] = false
using Altro
Altro.USE_OCTAVIAN

@time solver1 = ALTROSolver(Problems.Quadrotor(:zigzag)...)
@time solver2 = ALTROSolver(Problems.Quadrotor(:zigzag)...)
@time solve!(solver1)
@time solve!(solver2)