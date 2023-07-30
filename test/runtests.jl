using DifferentiableFactorizations,
    Test, Zygote, FiniteDifferences, LinearAlgebra, ComponentArrays, Random
Random.seed!(1)

const nreps = 3
const tol = 1e-8

@testset "Cholesky" begin
    for _ = 1:nreps
        A = rand(3, 3)

        f1(A) = diff_cholesky(A' * A + 2I).U
        zjac1 = Zygote.jacobian(f1, A)[1]
        fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
        @test norm(zjac1 - fjac1) < tol

        f2(A) = diff_cholesky(A' * A + 2I).L
        zjac2 = Zygote.jacobian(f2, A)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
        @test norm(zjac2 - fjac2) < tol
    end
end

@testset "LU" begin
    for _ = 1:nreps
        A = rand(3, 3)

        f1(A) = vec(diff_lu(A).U)
        zjac1 = Zygote.jacobian(f1, A)[1]
        fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
        @test norm(zjac1 - fjac1) < tol

        f2(A) = vec(diff_lu(A).L)
        zjac2 = Zygote.jacobian(f2, A)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
        @test norm(zjac2 - fjac2) < tol
    end
end

@testset "QR" begin
    for _ = 1:nreps
        A = rand(3, 2)

        f1(A) = vec(diff_qr(A).Q)
        zjac1 = Zygote.jacobian(f1, A)[1]
        fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
        @test norm(zjac1 - fjac1) < tol

        f2(A) = vec(diff_qr(A).R)
        zjac2 = Zygote.jacobian(f2, A)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
        @test norm(zjac2 - fjac2) < tol
    end
end

@testset "Eigen" begin
    for _ = 1:nreps
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
        @test norm(zjac1 - fjac1) < tol

        f2(AB) = begin
            A = AB.A' * AB.A
            B = AB.B' * AB.B + 2I
            vec(diff_eigen(A, B).V)
        end
        zjac2 = Zygote.jacobian(f2, AB)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, AB)[1]
        @test norm(zjac2 - fjac2) < tol

        f3(A) = diff_eigen(A' * A).s
        zjac3 = Zygote.jacobian(f3, A)[1]
        fjac3 = FiniteDifferences.jacobian(central_fdm(5, 1), f3, A)[1]
        @test norm(zjac3 - fjac3) < tol

        # Seems eigen does not guarantee differentiability of the output V without matrix B - the FiniteDifferences jacobian has large numbers

        # f4(A) = vec(diff_eigen(A' * A + 5I).V)
        # zjac4 = Zygote.jacobian(f4, A)[1]
        # fjac4 = FiniteDifferences.jacobian(central_fdm(5, 1), f4, A)[1]
        # @test norm(zjac4 - fjac4) < tol
    end
end

@testset "SVD" begin
    for _ = 1:nreps
        A = rand(3, 3)

        f1(A) = diff_svd(A).S
        zjac1 = Zygote.jacobian(f1, A)[1]
        fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
        @test norm(zjac1 - fjac1) < tol

        f2(A) = vec(diff_svd(A).U)
        zjac2 = Zygote.jacobian(f2, A)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
        @test norm(zjac2 - fjac2) < tol

        f3(A) = vec(diff_svd(A).V)
        zjac3 = Zygote.jacobian(f3, A)[1]
        fjac3 = FiniteDifferences.jacobian(central_fdm(5, 1), f3, A)[1]
        @test norm(zjac3 - fjac3) < tol
    end
end

@testset "Schur" begin
    for _ = 1:nreps
        A = randn(3, 3)
        A = A' + A + I
        f1(A) = vec(diff_schur(A).Z)
        zjac1 = Zygote.jacobian(f1, A)[1]
        fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, A)[1]
        @test norm(zjac1 - fjac1) < tol

        f2(A) = vec(diff_schur(A).T)
        zjac2 = Zygote.jacobian(f2, A)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, A)[1]
        @test norm(zjac2 - fjac2) < tol
    end
end

@testset "Generalized Schur" begin
    for _ = 1:nreps
        A = randn(3, 3)
        A = A' + A + I
        B = rand(3, 3)
        B = B' + B + I
        AB = ComponentVector(; A, B)

        f1(AB) = vec(diff_schur(AB.A, AB.B).left)
        zjac1 = Zygote.jacobian(f1, AB)[1]
        fjac1 = FiniteDifferences.jacobian(central_fdm(5, 1), f1, AB)[1]
        @test norm(zjac1 - fjac1) < tol

        f2(AB) = vec(diff_schur(AB.A, AB.B).right)
        zjac2 = Zygote.jacobian(f2, AB)[1]
        fjac2 = FiniteDifferences.jacobian(central_fdm(5, 1), f2, AB)[1]
        @test norm(zjac2 - fjac2) < tol

        f3(AB) = vec(diff_schur(AB.A, AB.B).S)
        zjac3 = Zygote.jacobian(f3, AB)[1]
        fjac3 = FiniteDifferences.jacobian(central_fdm(5, 1), f3, AB)[1]
        @test norm(zjac3 - fjac3) < tol

        f4(AB) = vec(diff_schur(AB.A, AB.B).T)
        zjac4 = Zygote.jacobian(f4, AB)[1]
        fjac4 = FiniteDifferences.jacobian(central_fdm(5, 1), f4, AB)[1]
        @test norm(zjac4 - fjac4) < tol
    end
end
