import LinearAlgebra: Factorization, Cholesky
import SparseArrays.SparseMatrixCSC
abstract type AbstractKLFactorization{Tv}<:Factorization{Tv} end
# An "Explicit" Cholesky factorization of a kernel matrix.

# An implicit Cholesky factorization of a kernel matrix that allows to perform prediction and compute the likelihood without explicitly storing the matrix
struct ImplicitKLFactorization{Tv,Ti,Tm,Tc<:AbstractCovarianceFunction{Tv}}<:AbstractKLFactorization{Tv}
    # Ordering
    P::Vector{Ti}
    # Skeletons that describe the sparsity pattern
    supernodes::IndirectSupernodalAssignment{Ti,Tm}
    # A covariance function
    𝒢::Tc
end

struct ExplicitKLFactorization{Tv,Ti,Tm,Tc}<:AbstractKLFactorization{Tv}
    # Ordering 
    P::Vector{Ti}
    # A list of the measurements that can be used to compute covariances with new data points
    measurements::Vector{Tm}
    # A covariance function
    𝒢::Tc
    # (Inverse-)Cholesky factor
    U::SparseMatrixCSC{Tv,Ti}
end 

function ExplicitKLFactorization(in::ImplicitKLFactorization{Tv,Ti,Tm,Tc}) where {Tv,Ti,Tm,Tc}
    return ExplicitKLFactorization{Tv,Ti,Tm,Tc}(in.P, in.supernodes.measurements, 𝒢, factorize(in.𝒢, in.supernodes))
end

# Construct an implicit KL Factorization 
# using 1-maximin and a single set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, supernodes, 𝒢)
end

# using k-maximin and a single set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, supernodes, 𝒢)
end

# using 1-maximin and multiple set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, supernodes, 𝒢)
end

# using k-maximin and multiple set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    # obtain measurements by concatenation
    measurements = reduce(vcat, collect.(measurements))
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, supernodes, 𝒢)
end

# Construct an implicit KL Factorization 
# using 1-maximin and a single set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# using k-maximin and a single set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# using 1-maximin and multiple set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# using k-maximin and multiple set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    # obtain measurements by concatenation
    measurements = reduce(vcat, collect.(measurements))
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,Tm,typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end



# The dense, exact Cholesky factorization. Only for debugging purposes.
struct DenseCholeskyFactorization{Tv,Tm,Tc}<:AbstractKLFactorization{Tv}
    L::Cholesky{Tv,Matrix{Tv}}
    measurements::Tm
    𝒢
end