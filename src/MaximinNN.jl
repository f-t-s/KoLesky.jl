# The functions in this file implement computation of the maximin ordering and sparsity pattern using the range search of NearestNeighbors.jl
import NearestNeighbors: KDTree, BallTree, inrange, nn, knn
import Base: sort!, sortperm, issorted

# function to update the heap in the case of ordinary distance
function _update_distances!(nearest_distances::AbstractVector, id, new_distance)
    nearest_distances[id] = min(nearest_distances[id], new_distance)
    return new_distance
end

# function to update the heap in the case of k-nearest neighbors 
# The largest distance, and hence the distance used for the  length scale, is the first entry of the corresponding column
function _update_distances!(nearest_distances::AbstractMatrix, id, new_distance)
    k_neighbors = size(nearest_distances, 1)
    # If the new distance is larger than the largest one in the stack, return the largest distance 
    if new_distance >= nearest_distances[1, id]
        return nearest_distances[1, id]
    else
        for k = 2 : k_neighbors
            # if k - 1 is the correct place for the new distance, insert it and return new first distance
            if nearest_distances[k, id] <= new_distance
                nearest_distances[k - 1, id] = new_distance
                return nearest_distances[1, id]
            # if this is not the case, move up the distance from position k to position k-1
            else 
                nearest_distances[k - 1, id] = nearest_distances[k, id]
            end
        end
        # if the new distance hasn't stored yet (this is the only condition under which the function has not returned yet), 
        # store it in the last place as the smalles recorded distance.
        nearest_distances[k_neighbors, id] = new_distance
        return nearest_distances[1, id]
    end
end

# including x, tree_function
function maximin_ordering(x::AbstractMatrix, k_neighbors; init_distances=fill(typemax(eltype(x)), (k_neighbors, size(x, 2))), Tree=KDTree)
    # constructing the tree
    N = size(x, 2)
    tree = Tree(x)
    nearest_distances= copy(init_distances)
    @assert size(nearest_distances) == (k_neighbors, N)
    for k = 1 : N
        sort!(vec(view(nearest_distances, :, k)); rev=true) 
    end
    heap = MutableHeap(vec(nearest_distances[1, :]))
    ℓ = Vector{eltype(init_distances)}(undef, N)
    P = Vector{Int64}(undef, N)
    for k = 1 : N 
        pivot = top_node!(heap)
        ℓ[k] = getval(pivot)
        P[k] = getid(pivot)
        # a little clunky, since inrange doesn't have an option to return range and we want to avoid introducing a 
        # distance measure separate from the NearestNeighbors
        number_in_range = length(inrange(tree, x[:, P[k]], ℓ[k]))            
        ids, dists = knn(tree, x[:, P[k]], number_in_range)
        for (id, dist) in zip(ids, dists)
            if id != getid(pivot)
                # update the distance as stored in nearest_distances
                new_dist = _update_distances!(nearest_distances, id, dist)
                # decreases the distance as stored in the heap
                update!(heap, id, new_dist)
            end
        end
    end
    # returns the maximin ordering P together with the distance vector. 
    return P, ℓ
end

# including x, tree_function
function maximin_ordering(x::AbstractMatrix; init_distances=fill(typemax(eltype(x)), (size(x, 2))), Tree=KDTree)
    # constructing the tree
    N = size(x, 2)
    tree = Tree(x)
    nearest_distances= copy(init_distances)
    @assert length(nearest_distances) == N
    heap = MutableHeap(nearest_distances)
    ℓ = Vector{eltype(init_distances)}(undef, N)
    P = Vector{Int64}(undef, N)
    for k = 1 : N 
        pivot = top_node!(heap)
        ℓ[k] = getval(pivot)
        P[k] = getid(pivot)
        # a little clunky, since inrange doesn't have an option to return range and we want to avoid introducing a 
        # distance measure separate from the NearestNeighbors
        number_in_range = length(inrange(tree, x[:, P[k]], ℓ[k]))            
        ids, dists = knn(tree, x[:, P[k]], number_in_range)
        for (id, dist) in zip(ids, dists)
            if id != getid(pivot)
                # update the distance as stored in nearest_distances
                new_dist = _update_distances!(nearest_distances, id, dist)
                # decreases the distance as stored in the heap
                update!(heap, id, new_dist)
            end
        end
    end
    # returns the maximin ordering P together with the distance vector. 
    return P, ℓ
end

# function to compute the distance of each element to the nearest element in the tree
function _construct_initial_distance(tree::Union{KDTree,BallTree}, x)
    return nn(tree, x)[2]
end

function _construct_initial_distance(tree::Union{KDTree,BallTree}, x, k_neighbors)
    if length(tree.data) >= k_neighbors
        # take the sets of knns, sort them, and concatenate them horizontally
        return hcat(sort!.(knn(tree, x, k_neighbors)[2], rev=true)...)
    else
        # if the tree is too small for the required number of neighbors (should not really happen)
        # Perform knn only on the largest admissible k
        k_max = length(tree.data)
        out_distances = hcat(sort!.(knn(tree, x, k_max)[2], rev=true)...)
        # Then padd remaining (leading) rows with typemax
        return vcat(fill(typemax(eltype(out_distances)), (k_neighbors - k_max)), out_distances)
    end
end

# combines multiple orderings of number 1 : Nₖ (represented by arrays Pₖ containing the indices in the order in which they appear)
# to an ordering of the set 1 : ∑ Nₖ
function concatenate_ordering(P_vec::AbstractVector{<:AbstractVector{<:Integer}})
    offset = 0
    P_out = eltype(eltype(P_vec))[]
    for P in P_vec
        append!(P_out, P .+ offset)
        offset += length(P)
    end
    return P_out
end 

# A maximin ordering that forces the point sets x ∈ x_vec to be ordered in the order in which they appear in x_vec. 
function maximin_ordering(x::AbstractVector{<:AbstractMatrix}; init_distances=[fill(typemax(eltype(eltype(x))), (size(xₖ, 2))) for xₖ in x], Tree=KDTree)

    for (k, xₖ) in enumerate(x)
        tree = Tree(xₖ)
        for l in (k + 1) : length(x)
            init_distances[l] .= min.(init_distances, _construct_initial_distance(tree, x))
        end
    end

    P = Vector{Int}[]
    ℓ = Vector{eltype(eltype(x))}[]
    for (xₖ, k) in enumerate(x)
        # create the ordering of k-th set of points
        Pₖ, ℓₖ = maximin_ordering(xₖ; init_distances=init_distances[k])
        push!(P, Pₖ)
        push!(ℓ, ℓₖ)
    end
    return concatenate_ordering(P), vcat(ℓ...)
end

# A mehtod of the maximin ordering that forces the point sets x ∈ x_vec to be ordered in the order in which they appear in x_vec. 
function maximin_ordering(x::AbstractVector{<:AbstractMatrix}, k_neighbors; init_distances=[fill(typemax(eltype(eltype(x))), (k_neighbors, size(xₖ, 2))) for xₖ in x], Tree=KDTree)

    for (k, xₖ) in enumerate(x)
        tree = Tree(xₖ)
        for l in (k + 1) : length(x)
            init_distances[l] .= min.(init_distances, _construct_initial_distance(tree, x, k_neighbors))
        end
    end

    P = Vector{Int}[]
    ℓ = Vector{eltype(eltype(x))}[]
    for (xₖ, k) in enumerate(x)
        # create the ordering of k-th set of points
        Pₖ, ℓₖ = maximin_ordering(xₖ, k_neighbors; init_distances=init_distances[k])
        push!(P, Pₖ)
        push!(ℓ, ℓₖ)
    end
    return concatenate_ordering(P), vcat(ℓ...)
end

# splits a given parent_list with parent scales ordered from coarse(large indices) to fine (small indices)
function _split_into_supernodes(parent_list, ℓ, λ)
    out = Vector{Int}[]
    ℓ_max = Inf
    node = Int[]
    for id in parent_list 
        # if given is large enough, add to present supernode 
        if ℓ[id] > ℓ_max / λ
            push!(node, id)
        elseif isempty(node)
            ℓ_max = ℓ[id]
        else             
            ℓ_max = ℓ[id]
            push!(out, node)
            node = Int[]
        end
    end
    # if there is a remaining node at the end, add it to the output
    isempty(node) || push!(out, node)
    return out
end

function _gather_assignments(assignments, first_parent) 
    perm = sortperm(assignments) 

    first_indices = unique(i -> assignments[perm[i]], 1 : length(perm))
    push!(first_indices, length(assignments) + 1) 
    ranges = [(first_indices[k] : (first_indices[k + 1] - 1)) for k = 1 : (length(first_indices) - 1)]
    return [perm[range] .+ (first_parent- 1) for range in ranges] 
end

# taking as input the maximin ordering and the associated distances, computes the associated reverse maximin sparsity pattern
# α determines what part of the sparsity pattern arises from the clustering as opposed to the the sparsity pattern of individual points. 
function supernodal_reverse_maximin_sparsity_pattern(x::AbstractMatrix, P, ℓ, ρ; lambda=1.5, alpha=1.0, Tree=KDTree, reconstruct_ordering=true)
    # want to avoid user facing unicode 
    λ = lambda
    α = alpha
    @assert λ > 1.0
    @assert 0.0 <= α <= 1.0
    @assert α * ρ > 1
    # constructing the tree
    N = size(x, 2)
    @assert N == length(P)
    # reordering x according to the reverse maximin ordering $P$ 
    # we will not use the original x. Note that this is a little wasteful for large size(x, 1). 
    x = x[:, P]

    # constructing a maximin ordering that are used as centers of the maximin ordering. 
    if reconstruct_ordering == true 
        P_temp, ℓ_temp = maximin_ordering(x; Tree)
        rev_P_temp = Vector{Int}(undef, N)
        rev_P_temp[P_temp] = 1 : N
    else
        P_temp = copy(P)
        ℓ_temp = copy(ℓ)
        rev_P_temp = Vector{Int}(undef, N)
        rev_P_temp[P_temp] = 1 : N
    end

    supernodes = IndexSuperNode{Int}[]
    children_tree = Tree(x) 
    min_ℓ = ℓ[findfirst(!isinf, ℓ)]
    last_aggregation_point = 1
    last_parent = 0
    # last_parent = findnext(l -> ℓ[l] < min_ℓ / λ, last_parent) - 1 
    while last_parent < N
        # finding the last aggregation index, for which the aggregaton points are sufficiently spread out away from each other
        last_aggregation_point = findnext(l -> (l == N + 1) || (ℓ_temp[l] < α * ρ * min_ℓ), 1 : (N + 1), last_aggregation_point) - 1
        # Constructing the aggregation tree containing only the admissible aggregation points
        aggregation_tree = Tree(x[:, P_temp[1 : last_aggregation_point]])

        # The first parent that we are treating in the present iteration of the while loop is first_parent
        first_parent = last_parent + 1
        # finding the last index l for which ℓ[l] is still within the admissible scale range
        last_parent = findnext(l -> (l + 1 == N + 1) || ℓ[l + 1] < min_ℓ, 1 : N, first_parent)

        # Computing the assignments to supernodes
        assignments = nn(aggregation_tree, x[:, first_parent : last_parent])[1]
        column_indices_list = _gather_assignments(assignments, first_parent)

        for column_indices in column_indices_list
            row_indices = Int[]
            for column_index in column_indices
                # possibly use second parameter here or make dependent on α
                new_row_indices = inrange(children_tree, x[:, column_index], ρ * ℓ[column_index])
                new_row_indices = new_row_indices[findall(new_row_indices .<= column_index)]
                append!(row_indices, new_row_indices)
            end
            sort!(row_indices)
            unique!(row_indices)
            push!(supernodes, IndexSuperNode(column_indices, row_indices))
        end

        # updating the length scale 
        min_ℓ = min_ℓ / λ
    end
    return supernodes
end

# high-level driver routine for creating the supernodal ordering and sparisty pattern
# Methods using 1-maximin ordering
function ordering_and_sparsity_pattern(x::AbstractMatrix, ρ; init_distances=fill(typemax(eltype(x)), (size(x, 2))), lambda=1.5, alpha=1.0, Tree=KDTree)
    P, ℓ = maximin_ordering(x; init_distances, Tree)
    supernodes = supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, ρ; lambda, alpha, Tree)
    return P, ℓ, supernodes
end 

function ordering_and_sparsity_pattern(x::AbstractVector{<:AbstractMatrix}, ρ; init_distances=init_distances=[fill(typemax(eltype(eltype(x))), (size(xₖ, 2))) for xₖ in x], lambda=1.5, alpha=1.0, Tree=KDTree)
    P, ℓ = maximin_ordering(x; init_distances, Tree)
    supernodes = supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, ρ; lambda, alpha, Tree)
    return P, ℓ, supernodes
end 

# Methods using k-maximin ordering
function ordering_and_sparsity_pattern(x::AbstractMatrix, k_neighbors, ρ; init_distances=fill(typemax(eltype(x)), (k_neighbors, size(x, 2))), lambda=1.5, alpha=1.0, Tree=KDTree)
    P, ℓ = maximin_ordering(x, k_neighbors; init_distances, Tree)
    supernodes = supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, ρ; lambda, alpha, Tree)
    return P, ℓ, supernodes
end 

function ordering_and_sparsity_pattern(x::AbstractVector{<:AbstractMatrix}, k_neighbors, ρ; init_distances=init_distances=[fill(typemax(eltype(eltype(x))), (k_neighbors, size(xₖ, 2))) for xₖ in x], lambda=1.5, alpha=1.0, Tree=KDTree)
    P, ℓ = maximin_ordering(x, k_neighbors; init_distances, Tree)
    supernodes = supernodal_reverse_maximin_sparsity_pattern(x, P, ℓ, ρ; lambda, alpha, Tree)
    return P, ℓ, supernodes
end 