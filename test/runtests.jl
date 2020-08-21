include("test_utils.jl")
using KoLesky
using Test
import KoLesky.distance

dofs = mat2points(rand(3, 10))
mindist = minimum([distance(dofs[k],dofs[l]) for k in 1 : length(dofs) for l in 1 : length(dofs) if k != l])
diam = maximum([distance(dofs[k],dofs[l]) for k in 1 : length(dofs) for l in 1 : length(dofs)])
sort_distance = sort([distance(dofs[k],dofs[l]) for k in 1 : length(dofs) for l in 1 : length(dofs) if k < l])
r = sqrt(minimum(sort_distance[2:end] ./ sort_distance[1 : (end - 1)]))

n_bins = 2 * round(Int, log(r, diam / mindist))

# r_P, r_ℓ = r_maximin(dofs; r=r, n_bins=n_bins)
# P, ℓ = KoLesky.naive_maximin(dofs)


# r_P, r_ℓ = r_maximin(dofs; r=r, n_bins=n_bins)
# P, ℓ = KoLesky.naive_maximin(dofs)

@testset "KoLesky.jl" begin
    # Write your tests here.
    @test r_maximin(dofs, r=r, n_bins=n_bins)[1] == KoLesky.naive_maximin(dofs)[1]
    @test r_maximin(dofs, r=r, n_bins=n_bins)[2] ≈ KoLesky.naive_maximin(dofs)[2]

    @test maximin(dofs,)[1] == KoLesky.naive_maximin(dofs)[1]
    @test maximin(dofs,)[2] ≈ KoLesky.naive_maximin(dofs)[2]
end
