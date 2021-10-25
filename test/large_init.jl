# Create a relatively large problem
N = 101
tf = 5.0
dt = tf / (N - 1)
model = RobotZoo.DoubleIntegrator(32)
n,m = size(model)
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1e-2, m))
Qf = Q * (N - 1)
xf = @SVector fill(10.0, n)
x0 = @SVector zeros(n)

obj = LQRObjective(Q, R, Qf, xf, N) 
prob = Problem(model, obj, xf, tf, x0 = x0)

##
t = @elapsed ALTROSolver(prob)
@test t < 60  # it should finish initializing the solver in under a minute (usually about 8 seconds on a desktop)