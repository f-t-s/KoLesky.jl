using KoLesky
import StaticArrays.SVector
using LinearAlgebra

N = 1000
d = 2
x = rand(d, N)
Δδ_coefs = -1.0
δ_coefs = 1.0
meas_δ = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(x[:,i])) for i = 1:N] 
meas_Δδ = [KoLesky.ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(x[:,i]), Δδ_coefs, δ_coefs) for i = 1:N]

measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
measurements[1] = meas_Δδ
measurements[2] = meas_δ

lengthscale = 0.2
cov = KoLesky.MaternCovariance7_2(lengthscale)

ρ = 12.0
implicit_factor = KoLesky.ImplicitKLFactorization(cov, measurements, ρ)
@time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
U = explicit_factor.U
P = explicit_factor.P
U_inv = inv(Matrix(U))
Theta_FC = U_inv'*U_inv

Theta = zeros(2*N,2*N)
cov(view(Theta,1:N,1:N),meas_Δδ)
cov(view(Theta,1:N, N+1:2*N),meas_Δδ,meas_δ)
view(Theta,N+1:2*N, 1:N)[:,:] = Theta[1:N, N+1:2*N]'
cov(view(Theta,N+1:2*N,N+1:2*N),meas_δ)

@show norm(Theta[P,P]-Theta_FC)/norm(Theta[P,P])
@show norm(Theta[P[1:N],P[1:N]]-Theta_FC[1:N,1:N])/norm(Theta[P[1:N],P[1:N]])
@show norm(Theta[P[N+1:2*N],P[N+1:2*N]]-Theta_FC[N+1:2*N,N+1:2*N])/norm(Theta[P[N+1:2*N],P[N+1:2*N]]);
