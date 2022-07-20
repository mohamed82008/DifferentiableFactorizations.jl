using DifferentiableFactorizations, Test, Zygote, FiniteDifferences, LinearAlgebra, ComponentArrays

@testset "Cholesky" begin
    A = rand(3, 3)

    f1(A) = diff_cholesky(A' * A + 2I).U
    zjac1 = Zygote.jacobian(f1, A)[1]
    fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
    @test norm(zjac1 - fjac1) < 1e-9

    f2(A) = diff_cholesky(A' * A + 2I).L
    zjac2 = Zygote.jacobian(f2, A)[1]
    fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
    @test norm(zjac2 - fjac2) < 1e-9
end

@testset "LU" begin
    A = rand(3, 3)

    f1(A) = vec(diff_lu(A).U)
    zjac1 = Zygote.jacobian(f1, A)[1]
    fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
    @test norm(zjac1 - fjac1) < 1e-9

    f2(A) = vec(diff_lu(A).L)
    zjac2 = Zygote.jacobian(f2, A)[1]
    fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
    @test norm(zjac2 - fjac2) < 1e-9
end

@testset "QR" begin
    A = rand(3, 2)
    
    f1(A) = vec(diff_qr(A).Q)
    zjac1 = Zygote.jacobian(f1, A)[1]
    fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
    @test norm(zjac1 - fjac1) < 1e-9

    f2(A) = vec(diff_qr(A).R)
    zjac2 = Zygote.jacobian(f2, A)[1]
    fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
    @test norm(zjac2 - fjac2) < 1e-9
end

@testset "Eigen" begin
    A = rand(3, 3)
    B = rand(3, 3)
    AB = ComponentVector(; A, B)

    f1(AB) = begin
        A = AB.A' * AB.A
        B = AB.B' * AB.B + 2I
        diff_eigen(A, B).s
    end
    zjac1 = Zygote.jacobian(f1, AB)[1]
    fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, AB)[1]
    @test norm(zjac1 - fjac1) < 1e-9

    f2(AB) = begin
        A = AB.A' * AB.A
        B = AB.B' * AB.B + 2I
        vec(diff_eigen(A, B).V)
    end
    zjac2 = Zygote.jacobian(f2, AB)[1]
    fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, AB)[1]
    @test norm(zjac2 - fjac2) < 1e-9

    f3(A) = diff_eigen(A' * A).s
    zjac3 = Zygote.jacobian(f3, A)[1]
    fjac3 = FiniteDifferences.jacobian(central_fdm(5, 1), f3, A)[1]
    @test norm(zjac3 - fjac3) < 1e-9

    # Seems eigen does not guarantee differentiability of the output V without matrix B - the FiniteDifferences jacobian has large numbers

    # f4(A) = vec(diff_eigen(A' * A + 5I).V)
    # zjac4 = Zygote.jacobian(f4, A)[1]
    # fjac4 = FiniteDifferences.jacobian(central_fdm(5, 1), f4, A)[1]
    # @test norm(zjac4 - fjac4) < 1e-9
end

@testset "SVD" begin
    A = rand(3, 3)

    f1(A) = diff_svd(A).S
    zjac1 = Zygote.jacobian(f1, A)[1]
    fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
    @test norm(zjac1 - fjac1) < 1e-9

    f2(A) = vec(diff_svd(A).U)
    zjac2 = Zygote.jacobian(f2, A)[1]
    fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
    @test norm(zjac2 - fjac2) < 1e-9

    f3(A) = vec(diff_svd(A).V)
    zjac3 = Zygote.jacobian(f3, A)[1]
    fjac3 = FiniteDifferences.jacobian(central_fdm(5, 1), f3, A)[1]
    @test norm(zjac3 - fjac3) < 1e-9
end
