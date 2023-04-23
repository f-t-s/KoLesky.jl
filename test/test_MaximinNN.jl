@testset "Supernodal Aggregation" begin
    @testset "_gather_assignment" begin
        in = [1; 1; 3; 1; 1; 5; 9; 9; 6; 7; 7; 7; 7;]
        out = [[[1; 2; 4; 5;]]; [[3;]]; [[6;]]; [[9;]]; [[10; 11; 12; 13]]; [[7;8;]];]
        @test KoLesky._gather_assignments(in, 1) == out
    end
end