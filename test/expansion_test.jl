using Test
using Altro
using RobotDynamics
using RobotZoo
const RD = RobotDynamics

# for T in (Float32,Float64)
@testset "Expansions ($T)" for T in (Float32, Float64)
    n,m = 10,5

    @testset "Dynamics expansion ($(typeof(model))" for model in (RobotZoo.Cartpole(), RobotZoo.Quadrotor())
        N = 11
        dt = 0.1
        model = RobotZoo.Quadrotor()
        dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
        n,m = RD.dims(model)
        e = RD.errstate_dim(model)

        # State Expansion
        E = Altro.StateControlExpansion{T}(e,m)
        @test size(E.xx) == (e,e)
        @test size(E.ux) == (m,e)
        @test size(E.uu) == (m,m)
        @test size(E.x) == (e,)
        @test size(E.u) == (m,)
        @test eltype(E.data) == T

        Eerr = Altro.CostExpansion{T}(e,m,N)
        Efull = Altro.FullStateExpansion(Eerr, dmodel)
        if model isa RobotZoo.Cartpole
            @test Efull === E
        end
        let E = Efull[1]
            @test size(E.xx) == (n,n)
            @test size(E.ux) == (m,n)
            @test size(E.uu) == (m,m)
            @test size(E.x) == (n,)
            @test size(E.u) == (m,)
            @test eltype(E.data) == T
        end

        S = Altro.StateControlExpansion{T}(n)
        @test size(S.xx) == (n,n)
        @test size(S.x) == (n,)
        @test_throws UndefRefError S.u
        @test eltype(S.data) == T

        Z = RD.Traj(map(1:N) do k
            x,u = rand(model)
            RD.KnotPoint{n,m}(Vector(x),Vector(u),(k-1)*dt,dt)
        end)
        D = [Altro.DynamicsExpansion{T}(n, e, m) for k = 1:N-1]

        for d in D
            if model isa RobotZoo.Cartpole
                @test d.∇f === d.∇e
            else
                @test d.∇f !== d.∇e
            end
        end

        sig = RD.StaticReturn()
        diffmethod = RD.ForwardAD()
        for k = 1:N-1
            RD.jacobian!(sig, diffmethod, dmodel, D[k], Z[k])
        end
        J = zeros(n,n+m)
        xn = zeros(n)
        for k = 1:N-1
            RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dmodel, J, xn, Z[k])
            @test J ≈ D[k].∇f
            eltype(D[1].A) == T
        end

        G = [zeros(T,n,e) for k = 1:N]
        Altro.errstate_jacobians!(dmodel,G,Z)
        for k = 1:N-1
            Gk = zeros(T,n,e)
            RD.state_diff_jacobian!(RD.RotationState(), model, Gk, Z[k])
            G[k] ≈ Gk
        end

        Altro.error_expansion!(dmodel, D, G)
        for k = 1:N-1
            A = D[k].A
            B = D[k].B
            G2 = G[k+1]
            G1 = G[k]
            Aerr = G2'A*G1
            Berr = G2'B
            @test D[k].fx ≈ Aerr
            @test D[k].fu ≈ Berr
        end
    end
end
