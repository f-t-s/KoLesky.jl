import SparseArrays: sparse, istriu
import Threads: @threads, threadid, nthreads
import LinearAlgebra: ldiv!
function _create_U_indices(supernodes::AbstractVector{<:IndexSuperNode{Ti}}) where Ti
    I = Ti[]
    J = Ti[]
    for node in supernodes, i in row_indices(node), j in column_indices(node)
        if i <= j 
            push!(I, i)
            push!(J, j)
        end
    end 
    return I, J
end

# computes the upper triangular 
import Threads: nthreads, @threads
function factorize(𝒢::AbstractCovarianceFunction{Tv}, supernodal_assignment::IndirectSupernodalAssignment{Ti,Tv}) where Tv
    # Determines the maximum amount of space needed 
    maximum_buffer_L_size = maximum((maximum.(size.(supernodal_assignment.supernodes)))^2)
    maximum_buffer_U_size = maximum(maximum.(size.(supernodal_assignment.supernodes, 1) .* size.(supernodal_assignment.supernodes, 2)))
    maximum_buffer_m_size = maximum(maximum.(size.(supernodal_assignment.supernodes, 1)))
    # The buffer used to store the lower triangular Cholesky factor that is computed as part of the update
    buffer_L = [Vector{Tv}(undef, maximum_buffer_L_size) for k = 1 : nthreads()]
    # The buffer used to store the columns, in order to to allow for a batched substitution
    buffer_U = [Vector{Tv}(undef, maximum_buffer_U_size) for k = 1 : nthreads()]
    # The buffer that is used to store the measurements of supernode
    buffer_m = [Vector{Tm}(undef, maximum_buffer_m_size) for k = 1 : nthreads()]
    I, J = _create_U_indices(supernodal_assignment.supernodes)
    @assert I == J
    # Create the upper triangular matrix U that will store the result
    U = sparse(I, J, Vector{Tv}(undef, length(I)))
    # Check that matrix is really upper triangular
    @assert istriu(U)

    @threads for node in supernodal_assignment.supernodes for i in row_indices(node) for j in column_indices(node)
        # extracting the thread's buffers
        n_rows = size(node, 1)
        n_columns = size(node, 2)
        buffer_L = reshape(view(buffer_L[threadid()], 1 : n_rows^2), n_rows, n_rows), 
        buffer_U = reshape(view(buffer_U[threadid()], 1 : n_rows * n_columns), n_rows, n_columns), 
        buffer_m = reshape(view(buffer_m[threadid()], 1 : n_rows))
        buffer_U .= 0
        for (j, i) in enumerate((n_rows - n_columns + 1) : n_rows)
            buffer_U[i, j] = 1
        end
        # Compute the local covariance Matrix 
        𝒢(buffer_L, buffer_m) 
        # Computing the Cholesky factorization
        chol = cholesky!(buffer_L)
        ldiv!(buffer_U, chol.U, buffer_U)
        # writing the results into the sparse matrix structure
        for (k, index) in enumerate(column_indices(node))
            U.nzvals[U.colptr[index] : (U.colptr[index + 1] - 1)] .= buffer_U[1 : (n_rows - n_columns + k),  k]
        end
    end
    return U
end