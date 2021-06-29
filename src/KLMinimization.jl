import SparseArrays: sparse, istriu
import Base.Threads: @threads, threadid, nthreads
import LinearAlgebra: ldiv!, cholesky!
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
import Base.Threads: nthreads, @threads
function factorize(𝒢::AbstractCovarianceFunction{Tv}, supernodal_assignment::IndirectSupernodalAssignment{Ti,Tm}) where {Tv,Ti,Tm}
    # Determines the maximum amount of space needed 
    maximum_buffer_L_size = maximum((maximum.(size.(supernodal_assignment.supernodes))))^2
    maximum_buffer_U_size = maximum(maximum.(size.(supernodal_assignment.supernodes, 1) .* size.(supernodal_assignment.supernodes, 2)))
    maximum_buffer_m_size = maximum(maximum.(size.(supernodal_assignment.supernodes, 1)))
    # The buffer used to store the lower triangular Cholesky factor that is computed as part of the update
    buffer_L = [Vector{Tv}(undef, maximum_buffer_L_size) for k = 1 : nthreads()]
    # The buffer used to store the columns, in order to to allow for a batched substitution
    buffer_U = [Vector{Tv}(undef, maximum_buffer_U_size) for k = 1 : nthreads()]
    # The buffer that is used to store the measurements of supernode
    buffer_m = [Vector{Tm}(undef, maximum_buffer_m_size) for k = 1 : nthreads()]
    I, J = _create_U_indices(supernodal_assignment.supernodes)
    @assert length(I) == length(J)
    # Create the upper triangular matrix U that will store the result
    U = sparse(I, J, Vector{Tv}(undef, length(I)))
    # Check that matrix is really upper triangular
    @assert istriu(U)

    # @threads for node in supernodal_assignment.supernodes
    for node in supernodal_assignment.supernodes
        # extracting the thread's buffers
        n_rows = size(node, 1)
        n_columns = size(node, 2)
        local_buffer_L = reshape(view(buffer_L[threadid()], 1 : n_rows^2), n_rows, n_rows) 
        local_buffer_U = reshape(view(buffer_U[threadid()], 1 : (n_rows * n_columns)), n_columns, n_rows) 
        local_buffer_U .= 0
        local_buffer_m = view(buffer_m[threadid()], 1 : n_rows)
        for (ind, i) in enumerate(column_indices(node))
            local_buffer_m[ind] = supernodal_assignment.measurements[i]
        end

        # Set the values of U to be a truncated identity matrix
        for (j, i) in enumerate((n_columns - n_rows + 1) : n_columns)
            local_buffer_U[i, j] = 1
        end
        # Compute the local covariance Matrix 
        𝒢(local_buffer_L, local_buffer_m) 
        # Computing the Cholesky factorization
        display(local_buffer_m)
        display(local_buffer_L)
        chol = cholesky!(local_buffer_L)
        ldiv!(local_buffer_U', chol.U, local_buffer_U)
        # writing the results into the sparse matrix structure
        for (k, index) in enumerate(column_indices(node))
            @show U.colptr[index]
            @show U.colptr[index + 1] - 1
            @show (n_columns - n_rows + k)
            U.nzval[U.colptr[index] : (U.colptr[index + 1] - 1)] .= local_buffer_U[1 : (n_columns - n_rows + k),  k]
        end
    end
    return U
end