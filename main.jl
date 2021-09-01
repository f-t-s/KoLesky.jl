using KoLesky
using Plots
using Random
using LinearAlgebra

Random.seed!(123)

# N = 10000
# x = rand(2, N) 

n = 100
h = 1 / (n + 1)
x = reduce(hcat, [[x,y] for x in h : h : (1 - h), y in h : h : (1 - h)])
N = n^2

# x_boundary = hcat(vcat(rand(1000)', rand([0.0, 1.0], 1000)'),
#                   vcat(rand([0.0, 1.0], 1000)', rand(1000)'))

x_boundary = hcat(vcat(ones(500)', Vector(0 : 1/500 : (1 - 1 / 500))'),
                  vcat(Vector(1 : -1/500 : 1/500)', ones(500)'),
                  vcat(Vector((1 - 1/500)  : -1/500 : 0)', zeros(500)'),
                  vcat(zeros(500)', Vector(1/500 : 1/500 : 1)'))


interior_measurements = KoLesky.point_measurements(x)
boundary_measurements = KoLesky.point_measurements(x_boundary)
N = N + length(boundary_measurements)
measurements = [boundary_measurements, interior_measurements]



𝒢 = KoLesky.MaternCovariance5_2(0.1)

implicit_factor = KoLesky.ImplicitKLFactorization(𝒢, measurements, 12.0,1)

@time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)

# comparing to true result

KM = zeros(N, N)
𝒢(KM, vcat(measurements...))

@show norm(KoLesky.assemble_covariance(explicit_factor) - KM ) / norm(KM)