using KoLesky
using Test

@testset "KoLesky.jl" begin
    # Write your tests here.
    @testset "MaximinNN.jl" begin
        include("test_MaximinNN.jl")
    end
end


