@testset "generators" begin
    M = rand(4, 2200)
    points = KoLesky.matrix2points(M)
    same = true
    for i = 1 : 4, j = 1 : 2200
        same &= (points[j].location[i] == M[i, j])
    end
    @test same
end
@testset "distances" begin
    import Distances: Euclidean, pairwise
    d = 5
    N1 = 2020
    N2 = 1783
    M1 = rand(5, N1) 
    M2 = rand(5, N2) 
    ps1 = KoLesky.matrix2points(M1)
    ps2 = KoLesky.matrix2points(M2)
    dst = zeros(N1, N2)
    for i = 1 : N1, j = 1 : N2
        dst[i, j] = KoLesky.euclidean_distance(ps1[i], ps2[j])
    end
    true_dst = pairwise(Euclidean(),M1, M2, dims=2)
    @test dst ≈ true_dst
end