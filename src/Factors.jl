import LinearAlgebra: Factorization, Cholesky
import SparseArrays.SparseMatrixCSC
abstract type AbstractKLFactorization{Tv}<:Factorization{Tv} end
# An "Explicit" Cholesky factorization of a kernel matrix.

# An implicit Cholesky factorization of a kernel matrix that allows to perform prediction and compute the likelihood without explicitly storing the matrix
struct ImplicitKLFactorization{Tv,Ti,Tm,Tc<:AbstractCovarianceFunction{Tv}}<:AbstractKLFactorization{Tv}
    # Ordering
    P::Vector{Ti}
    # Skeletons that describe the sparsity pattern
    # The measurements used by the Supernodal assignment are ordered # according to P
    supernodes::IndirectSupernodalAssignment{Ti,Tm}
    # A covariance function
    𝒢::Tc
end

struct ExplicitKLFactorization{Tv,Ti,Tm,Tc}<:AbstractKLFactorization{Tv}
    # Ordering 
    P::Vector{Ti}
    # A list of the measurements that can be used to compute covariances with new data points
    # The measurements are already ordered according to P
    measurements::Vector{Tm}
    # A covariance function
    𝒢::Tc
    # (Inverse-)Cholesky factor
    U::SparseMatrixCSC{Tv,Ti}
end 

function ExplicitKLFactorization(in::ImplicitKLFactorization{Tv,Ti,Tm,Tc}; nugget = 0.0) where {Tv,Ti,Tm,Tc}
    return ExplicitKLFactorization{Tv,Ti,Tm,Tc}(in.P, in.supernodes.measurements, in.𝒢, factorize(in.𝒢, in.supernodes; nugget = nugget))
end

# Construct an implicit KL Factorization 
# using 1-maximin and a single set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements))) 
    # get_coordinate gets the sparse array format, then collect gets the standard array format
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)[P]  # seems extra?
    supernodes = IndirectSupernodalAssignment(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, supernodes, 𝒢)
end

# using k-maximin and a single set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)[P]
    supernodes = IndirectSupernodalAssignment(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, supernodes, 𝒢)
end

# using 1-maximin and multiple set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = reduce(vcat, collect.(measurements))[P]
    supernodes = IndirectSupernodalAssignment(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, supernodes, 𝒢)
end

# using k-maximin and multiple set of measurments
function ImplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    # @show ℓ
    Ti = eltype(P)
    # obtain measurements by concatenation
    measurements = reduce(vcat, collect.(measurements))[P]
    supernodes = IndirectSupernodalAssignment(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, supernodes, 𝒢)
end

# using k-maximin and multiple set of measurments: specific to the case that ordering of derivative measurements follows that of diracs
function ImplicitKLFactorization_FollowDiracs(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # measurments[1] is Diracs on the boundary, measurements[2] Diracs on the interior
    lm = length(measurements)
    @assert lm >= 3
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : 2]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)

    # for the Diracs part
    # measurements_diracs = reduce(vcat, collect.(measurements[1:2]))[P]
    # supernodes_diracs = IndirectSupernodalAssignment(copy(supernodes), measurements_diracs)
    # ImplicitFactor_diracs = ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, supernodes_diracs, 𝒢)

    # obtain measurements by concatenation
    N_boundary = length(measurements[1])
    N_domain = length(measurements[2])
    P_all = [0 for _ in 1:(lm-1)*N_domain+N_boundary]
    P_all[1:N_boundary] = P[1:N_boundary]
    P_all[N_boundary+1:end] = reduce(vcat, [x.+ collect(0:lm-2)*N_domain for x in P[N_boundary+1:end]])
    measurements = reduce(vcat, collect.(measurements))[P_all]

    

    # construct supernodes
    for node in supernodes
        m = length(node.row_indices)
        n = length(node.column_indices)
        for i in 1:m
            rowi = node.row_indices[i]
            if rowi > N_boundary
                node.row_indices[i] = (rowi-N_boundary-1)*(lm-1) + N_boundary + 1
                append!(node.row_indices,(rowi-N_boundary-1)*(lm-1)+N_boundary+2:(rowi-N_boundary)*(lm-1)+N_boundary)
            end
        end
        sort!(node.row_indices)
        for j in 1:n
            columni = node.column_indices[j]
            if columni > N_boundary
                node.column_indices[j] = (columni-N_boundary-1)*(lm-1) + N_boundary + 1
                append!(node.column_indices,(columni-N_boundary-1)*(lm-1)+N_boundary+2:(columni-N_boundary)*(lm-1)+N_boundary)
            end
        end
        sort!(node.column_indices)
    end
    supernodes = IndirectSupernodalAssignment(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P_all, supernodes, 𝒢)
end


####
function ImplicitKLFactorization_DiracsFirstThenUnifScale(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree) where Tv
    # measurments[1] is Diracs on the boundary, measurements[2] Diracs on the interior
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern_DiracsFirstThenUnifScale(x, ρ, k_neighbors; lambda, alpha, Tree)

    Ti = eltype(P)
    # obtain measurements by concatenation
    measurements = reduce(vcat, collect.(measurements))[P]
    supernodes = IndirectSupernodalAssignment(supernodes, measurements)
    return ImplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, supernodes, 𝒢)
end

# Construct an implicit KL Factorization 
# using 1-maximin and a single set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree, nugget = 0.0) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)[P]
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# using k-maximin and a single set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractPointMeasurement}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree, nugget = 0.0) where Tv
    x = reduce(hcat, collect.(get_coordinate.(measurements)))
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)[P]
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# using 1-maximin and multiple set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ; lambda=1.5, alpha=1.0, Tree=KDTree, nugget = 0.0) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ; lambda, alpha, Tree)
    Ti = eltype(P)
    measurements = collect(measurements)[P]
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# using k-maximin and multiple set of measurments
function ExplicitKLFactorization(𝒢::AbstractCovarianceFunction{Tv}, measurements::AbstractVector{<:AbstractVector{<:AbstractPointMeasurement}}, ρ, k_neighbors; lambda=1.5, alpha=1.0, Tree=KDTree, nugget = 0.0) where Tv
    # x is now a vector of matrices
    x = [reduce(hcat, collect.(get_coordinate.(measurements[k]))) for k = 1 : length(measurements)]
    P, ℓ, supernodes = ordering_and_sparsity_pattern(x, ρ, k_neighbors; lambda, alpha, Tree)
    Ti = eltype(P)
    # obtain measurements by concatenation
    measurements = reduce(vcat, collect.(measurements))[P]
    supernodes = IndirectSupernodalAssignment{Ti}(supernodes, measurements)
    return ExplicitKLFactorization{Tv,Ti,eltype(measurements),typeof(𝒢)}(P, measurements, 𝒢, factorize(𝒢, supernodes))
end

# Assembling the approximate kernel matrix implied by a factorization
function assemble_covariance(factor::ExplicitKLFactorization)
    inv_U_matrix = inv(Matrix(factor.U))
    inv_P = similar(factor.P)
    inv_P[factor.P] = 1 : length(inv_P)

    return (inv_U_matrix' * inv_U_matrix)[inv_P, inv_P]
end 


# The dense, exact Cholesky factorization. Only for debugging purposes.
struct DenseCholeskyFactorization{Tv,Tm,Tc}<:AbstractKLFactorization{Tv}
    L::Cholesky{Tv,Matrix{Tv}}
    measurements::Tm
    𝒢
end