using KoLesky
using Test

@testset "KoLesky.jl" begin
    # Write your tests here.
    # @testset "Points.jl" begin
    #     include("test_Points.jl")
    # end
    @testset "MaximinNN.jl" begin
        include("test_MaximinNN.jl")
    end
end


