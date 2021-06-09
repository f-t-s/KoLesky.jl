import LinearAlgebra.Factorization
import SparseArrays.SparseMatrixCSC
abstract type AbstractKLFactorization{Tv}<:Factorization{Tv} end
# An "Explicit" Cholesky factorization of a kernel matrix.
struct ExplicitKLFactorization{Tv,Ti,Tm,Tc}<:AbstractKLFactorization{Tv}
    # Ordering 
    P::Vector{Ti}
    # A list of the measurements that can be used to compute covariances with new data points
    measurements::Vector{Tm}
    # A covariance function
    𝒢::Tc
    # (Inverse-)Cholesky factor
    L::SparseMatrixCSC{RT,Int}
end 

# An implicit Cholesky factorization of a kernel matrix that allows to perform prediction and compute the likelihood without explicitly storing the matrix
struct ImplicitKLFactorizationRBF{Tv,Ti,Tm,Tc}<:AbstractKLFactorization{Tv}
    # Ordering
    P::Vector{Ti}
    # Skeletons that describe the sparsity pattern
    supernodes::IndirectSupernodalAssignment{Ti,Tm}
    # A covariance function
    𝒢::Tc
end

# For now, we are inhereting the real types from the location vectors, and we assume the use of the euclidean distance, and an rbf
# function ImplicitKLFactorizationRBF(𝒢!, x::AbstractVector{<:Point}, ρ, λ)
#     P, ~, ~, skeletons = ordering_and_skeletons(x, ρ, λ)
#     return ImplicitKLFactorizationRBF{eltype(eltype(x)), eltype(x)}(P, skeletons, x, 𝒢!)
# end

