using KoLesky
using Test

@testset "KoLesky.jl" begin
    # Write your tests here.
    @testset "Points.jl" begin
        include("test_Points.jl")
    end
end


