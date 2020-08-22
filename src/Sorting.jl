include("./DOF.jl")
include("./Family.jl")
include("./MutableMaxHeap.jl")

# Revisit later 
# function r_maximin(dofs; r=1.1, n_bins=100, ℓ0=fill(Inf, length(dofs)))
#     # Initializing vector containing the reordering
#     P = Vector{Int}(undef, 0)
#     # Checklist to see which element has been added to orderigng already
#     checklist = trues(length(dofs))
#     # choosing the first pivot as having minimal distance according to the "hard coded" distance
#     push!(P, argmax(ℓ0))
#     # vector that stores the length scales of the dofs
#     ℓ_out = similar(ℓ0)
#     ℓ_out[P[1]] = ℓ0[P[1]]
#     checklist[P[1]] = false
#     # updating hte distances according to first pivot
#     ℓ = min.(distance.(dofs, Ref(dofs[P[1]])), ℓ0)
# 
#     # compute upper bound of diameter of domain
#     diam = 2 * maximum(ℓ) 
# 
#     f = Family(dofs)
#     # Setting the descendants of the first node to be all dofs, ordering the distance 
#     descendants(f)[P[1]] = sort([(ℓ[k], k) for k = 1 : length(ℓ)])
#     parents(f) .= [(ℓ[k], P[1]) for k = 1 : length(ℓ)]
# 
#     # instantiate rack 
#     rack = Rack(n_bins, r, diam, ℓ)
# 
#     # starting by looking in first bin
#     first_bin = 1
#     while ~iszero(first_bin)
#         # obtain new pivot
#         first_bin, pivot_id = next_dof(first_bin, rack)
#         # if pivot was not in last bin
# 
#         if !iszero(pivot_id)
#             new_pivot!(P, ℓ_out, rack, f, pivot_id, dofs, ℓ, checklist) 
#             checklist[pivot_id] = false
#         end
#     end
#     # Once all remaining dofs are in the smallest bin, we simply add them to the ordering in arbitrary order
#     # First set the length-scale of the remaining dofs to the minimal resolution
#     ℓ_out[setdiff(1 : length(dofs), P)] = ℓ[setdiff(1 : length(dofs), P)]
#     P = unique(vcat(P, 1 : length(dofs)))
#     return P, ℓ_out
# end

function new_pivot!(ℓ, h::MutableMaxHeap, f::Family, buffer, pivot, dofs)  
    # add pivot to ordering. Note that this tags the pivot and leads to it NOT being added automatically to its descendants.
    pivot_id, pivot_ℓ = pivot.id, pivot.val
    new_column!(f, pivot_id)
    pivot_number = f.n_columns[1]
    pivot = dofs[pivot_id]

    # add the pivot to it's own list of children 
    buffer_iterator = 1
    buffer[buffer_iterator] = Member(zero(typeof(pivot_ℓ)), pivot_id)
    # and downgrade its distance in the heap
    decrease_key!(h, pivot_id, 0.0)

    # load the distance from the pivot to its parent
    distance_pivot_parent = f.parents[pivot_id].val
    for index in column_iterator(f, f.parents[pivot_id].id)
        parent_descendant = f.rowval[index]
        distance_parent_descendant = parent_descendant.val

        # break loop if descendant is too far from parent to be descendant of pivot
        if distance_parent_descendant >  distance_pivot_parent + 2 * pivot_ℓ 
            break
        # else compute distance and possibly add as descendant or parent 
        elseif f.revP[parent_descendant.id] == zero(eltype(f.revP))
            distance_descendant_pivot = distance(dofs[parent_descendant.id], pivot)

            # If distance between pivot and descendant is small enough, add descendent to children of pivot
            # In this case, we also update the length scale of the pivot
            if distance_descendant_pivot < 2 * pivot_ℓ
                buffer_iterator += 1
                buffer[buffer_iterator] = Member(distance_descendant_pivot, parent_descendant.id)
                decrease_key!(h, parent_descendant.id, distance_descendant_pivot)
                ℓ[parent_descendant.id] = min(distance_descendant_pivot, ℓ[parent_descendant.id])
            end
            # if the descendant was not added to the ordering yet, update distance and check if pivot should become new parent of descendant.
            # Check if pivot should become the descendant's next parent 
            if distance_descendant_pivot + 2 * ℓ[parent_descendant.id] < pivot_ℓ && distance_descendant_pivot < distance_parent_descendant
                f.parents[parent_descendant.id] = Member(distance_descendant_pivot, pivot_number)
            end
        end
    end 
    # Sorts the descendants of the new pivot by distance to the pivot
    viewBuffer = view(buffer, 1 : buffer_iterator)
    sort!(viewBuffer, alg=QuickSort)
    new_children!(f, viewBuffer)
end

# computing the exact naive minimax ordering
function naive_maximin(dofs; ℓ0=fill(Inf, length(dofs)))
    P = Vector{Int}(undef, length(dofs))
    ℓ = Vector{Float64}(undef, length(dofs))
    P[1] = 1 
    ℓ[1] = Inf
    for k = 2 : length(dofs)
        dists = fill(Inf, length(dofs))
        for (l, distval) in enumerate(dists)
            for m = 1 : (k - 1)
                dists[l] = min(dists[l], distance(dofs[P[m]], dofs[l]))
            end
        end
        P[k] = argmax(dists)
        ℓ[P[k]] = maximum(dists)
    end
    return P, ℓ
end

function maximin(dofs; ℓ0=fill(Inf, length(dofs)))
    N = length(dofs)

    # instantiate the family and choose first pivot
    f = Family(dofs)
    new_column!(f, argmax(ℓ0))
    # updating the distances according to first pivot and add corresponding descendants to f
    ℓ = min.(distance.(dofs, Ref(dofs[f.P[1]])), ℓ0)
    new_children!(f, sort([Member(ℓ[id], id) for id = 1 : length(dofs)]))
    # setting all parents to be given by the first node
    f.parents .= [Member(ℓ[id], 1) for id = 1 : N]
    
    # Instantiate buffer to be used later.
    buffer = Vector{eltype(f.rowval)}(undef, length(dofs))
    # Instantiate the mutable binary heap with all distances set to maximum
    h = MutableMaxHeap{Float64,Int,Int}([Node{Float64,Int,Int}(typemax(Float64), i, zero(Int)) for i = 1 : length(dofs)], 1 : N)
    # adapt keys in heap
    for k = 1 : N 
        decrease_key!(h, k, ℓ[k])
    end
    # set the length scale of the first entry to the output value.
    ℓ[f.P[1]] = ℓ0[f.P[1]]

    # compute the remaining parts of the ordering. 
    # Right now the ordering over all dofs is computed but in the future we could add additional criteria for the breaking the for loop, for instance when pivot_ℓ falls below a treshhold.
    for k = 2 : length(ℓ)
        # obtain new pivot
        pivot = top_node!(h)
        
        # if pivot was not in last bin
        new_pivot!(ℓ, h, f, buffer, pivot, dofs)
    end
    # First set the length-scale of the remaining dofs to the minimal resolution
    return f.P, ℓ
end
