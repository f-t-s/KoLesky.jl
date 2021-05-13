# The functions in this file implement computation of the maximin ordering and sparsity pattern using the range search of NearestNeighbors.jl
import NearestNeighbors: KDTree, BallTree, inrange, nn, knn
import Base: sort!, sortperm, issorted

# function to update the heap in the case of ordinary distance
function _update_distances!(nearest_distances::AbstractVector, id, new_distance)
    nearest_distances[id] = min(nearest_distances[id], new_distance)
    return new_distance
end

# function to update the heap in the case of k-nearest neighbors 
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
function maximin_ordering(x::AbstractMatrix, k_neighbors; init_distances=fill(Inf, (k_neighbors, size(x, 2))), Tree=KDTree)
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
function maximin_ordering(x::AbstractMatrix; init_distances=fill(Inf, (size(x, 2))), Tree=KDTree)
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

# taking as input the maximin ordering and the associated distances, computes the associated reverse maximin sparsity pattern
# α determines what part of the sparsity pattern arises from the clustering as opposed to the the sparsity pattern of individual points. 
# TODO: Still bugged
function supernodal_reverse_maximin_sparsity_pattern(x::AbstractMatrix, P, ℓ, ρ, λ=1.5, α=1.0; Tree=KDTree, reconstruct_ordering=true)
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
        # TODO: replace with explicit 1-nn variant
        P_temp, ℓ_temp = maximin_ordering(x, 1; init_distances=fill(Inf, (1, size(x, 2))), Tree)
        rev_P_temp = Vector{Int}(undef, N)
        rev_P_temp[P_temp] = 1 : N
    else
        P_temp = copy(P)
        ℓ_temp = copy(ℓ)
        rev_P_temp = Vector{Int}(undef, N)
        rev_P_temp[P_temp] = 1 : N
    end

    tree = Tree(x)
    # nearest_distances= copy(init_distances)
    # @assert size(nearest_distances) == (k_neighbors, N)
    # iter_parent and child are used to find cutoff for when searching for the parent and child supernodes that a given 
    # point is part of
    iter_parent = 1

    # Initializing the lists of points that are associated to the different supernodes
    parent_lists = Vector{Vector{Int}}(undef, N)
    for k = 1 : N
        parent_lists[k] = Int[]
    end

    # We assign each point to the admissible (sufficiently large length scale compared to the sparsity pattern)
    # TODO check the choice of iter_parent regarding +- 1
    # In order to preserve the efficiency of the nearest neigbors, we repeatedly build th 
    tree = Tree(x[:,1:1])
    for k = 1 : N
        iter_parent = findnext(l -> ℓ_temp[l] < α * ρ * ℓ[k], 1 : N, iter_parent) - 1
        if  length(tree.data) < iter_parent / λ
            tree = Tree(x[:, 1 : min(end, round(Int, iter_parent * 1.5))])
        end
        # Computes, among all nodes before iter_parent or iter_child according to the ordering given by P_temp, the points closest to the given point x[:, k]
        # selects the list of indices (which here contains only one element) and then first element of this list
        parent_home = nn(tree, x[:, k], l -> rev_P_temp[l] > iter_parent)[1]
        push!(parent_lists[parent_home], k) 
    end
    # sorting each of the parent lists
    sort!.(parent_lists)

    # We now construct the parent sets while making sure that the ratio of length scales of parent nodes
    # within a given supernode is upper bounded by λ
    supernodal_parents_list = Vector{Int}[]
    for parent_list in parent_lists
        append!(supernodal_parents_list, _split_into_supernodes(parent_list, ℓ, λ))
    end

    # For now, we assemble the sparsity pattern by looking at patterns of the individual nodes
    supernodes = IndexSuperNode{Int}[]
    tree = Tree(x) 
    for parent_list in supernodal_parents_list
        children_list = Int[]
        for parent_id in parent_list
            new_children = inrange(tree, x[:, parent_id], ρ * ℓ[parent_id])
            new_children = new_children[findall(new_children .<= parent_id)]
            append!(children_list, new_children)
        end
        sort!(children_list)
        unique!(children_list)
        push!(supernodes, IndexSuperNode(parent_list, children_list)) 
    end
    return supernodes, supernodal_parents_list
end
