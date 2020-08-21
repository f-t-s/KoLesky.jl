include("./DOF.jl")
using DataStructures

# mapping a distance to a bin id
function dist2bin(ℓ, r, diam, n_bins)
    floor(Int, min(1 - log(r, ℓ / diam), n_bins))
end

# A struct that contains the assignment of the different points to the different bins of length scales
struct Rack
    bins::Vector{Vector{Int}}
    r::Float64
    diam::Float64
    ℓ::Vector{Float64}
    addresses::Vector{Tuple{Int,Int}}
end

# Constructor for Rack
function Rack(n_bins, r, diam, ℓ) 
    bins = Vector{Vector{Int}}(undef, n_bins)
    for i = 1 : length(bins)
        bins[i] = Vector{Int}(undef,0)
    end
    addresses = Vector{Tuple{Int,Int}}(undef, length(ℓ))
    for (i, l) in enumerate(ℓ)
        bin = dist2bin(l, r, diam, n_bins)
        push!(bins[bin], i)
        addresses[i] = (bin, length(bins[bin]))
    end
    return Rack(bins, r, diam, ℓ, addresses)
end 

# function that modifies distance value. Will only reduce the distance value and otherwise not to anything
function reduce_distance!(rack::Rack, id, l) 
    if l < rack.ℓ[id]
        # updating the distance
        rack.ℓ[id] = min(l, rack.ℓ[id])
        # finding the new bin
        new_bin = dist2bin(l, rack.r, rack.diam, length(rack.bins))
        # finding the old address (and therefore the old bin)
        old_address = rack.addresses[id]
        # only act, if the bin changes
        if new_bin > old_address[1]
            # replacing the old position of id with the last element of the bin
            rack.bins[old_address[1]][old_address[2]] = rack.bins[old_address[1]][end]
            # updating the address of the last element of the bin
            rack.addresses[rack.bins[old_address[1]][end]] = old_address
            # dropping the last element of the original bin
            pop!(rack.bins[old_address[1]])
            # add id to new bin
            push!(rack.bins[new_bin], id)
            # overwrite address of id
            rack.addresses[id] = (new_bin, length(rack.bins[new_bin]))
        end
    end
end

# looks for the next dof to include in the ordering, starting at the bin first_bin
# returns the new first_bin, where it found the dof, as well as the id of the dof.
# If all remaining dofs are in the last bin, it signals it by returning zero
function next_dof(first_bin::Int, rack::Rack)
    while first_bin < length(rack.bins)
        if !isempty(rack.bins[first_bin]) 
            return first_bin, first(rack.bins[first_bin])
        end
        first_bin += 1
    end
    return 0, 0
end

struct Family
    descendants::Vector{Vector{Tuple{Float64, Int}}}
    parents::Vector{Tuple{Float64, Int}}
    scratch::Vector{Tuple{Float64,Int}}
    function Family(descendants::Vector{Vector{Tuple{Float64, Int}}}, parents::Vector{Tuple{Float64, Int}})
        return new(descendants, parents, Vector{Tuple{Float64, Int}}(undef, length(parents)))
    end
end

function Family(dofs)
    N = length(dofs)
    descendants = Vector{Vector{Tuple{Float64, Int}}}(undef, N)
    parents = Vector{Tuple{Float64, Int}}(undef, N)
    for k = 1 : N
        descendants[k] = Vector{Tuple{Float64, Int}}(undef, 0)
    end
    return Family(descendants, parents)
end

function descendants(family::Family)
    return family.descendants
end

function close_descendants(id, l, family::Family)
    if getindex(descendants(family)[id][end], 1) > l
        last_ind = findfirst(x -> getindex(x, 1) > l, descendants(family)[id])
    else
        last_ind = length(descendants(family)[id])
    end
    return view(descendants(family)[id], 1 : last_ind)
end

function parents(family::Family)
    return family.parents
end

function new_pivot!(P, ℓ_out, rack::Rack, f::Family, pivot_id, dofs, ℓ, checklist) 
    # add pivot to ordering
    push!(P, pivot_id)
    pivot_ℓ = ℓ[pivot_id]
    pivot = dofs[pivot_id]
    pivot_parent = parents(f)[pivot_id]

    # remember the length scale for the output
    ℓ_out[pivot_id] = pivot_ℓ
    # Iterating over descendants of the parent that are close enough to the parent to possibly be descendants of the pivot
        # Update the distance
    # @show length(close_descendants(pivot_parent[2], distance(dofs[pivot_parent[2]], pivot) + 2 * pivot_ℓ, f))
    for parent_descendant in close_descendants(pivot_parent[2], distance(dofs[pivot_parent[2]], pivot) + 2 * pivot_ℓ, f)
        distance_descendant_pivot = distance(dofs[parent_descendant[2]], pivot)
        if checklist[parent_descendant[2]] == true 
            # Update the distance
            reduce_distance!(rack, parent_descendant[2], distance_descendant_pivot)
            # Check if distance is small enough for descendant to be included as descendant of the pivot
            if distance_descendant_pivot < 2 * pivot_ℓ
                push!(descendants(f)[pivot_id], (distance_descendant_pivot, parent_descendant[2]))
            end
            # Check if pivot should become the descendant's next parent 
            if distance_descendant_pivot + 2 * ℓ[parent_descendant[2]] < pivot_ℓ && distance_descendant_pivot < parents(f)[parent_descendant[2]][1]
                parents(f)[parent_descendant[2]] = (distance_descendant_pivot, pivot_id)
            end
        end
    end 
    # Sorts the descendants of the new pivot by distance to the pivot
    sort!(descendants(f)[pivot_id], alg=QuickSort)
end


function r_maximin(dofs; r=1.1, n_bins=100, ℓ0=fill(Inf, length(dofs)))
    # Initializing vector containing the reordering
    P = Vector{Int}(undef, 0)
    # Checklist to see which element has been added to orderigng already
    checklist = trues(length(dofs))
    # choosing the first pivot as having minimal distance according to the "hard coded" distance
    push!(P, argmax(ℓ0))
    # vector that stores the length scales of the dofs
    ℓ_out = similar(ℓ0)
    ℓ_out[P[1]] = ℓ0[P[1]]
    checklist[P[1]] = false
    # updating hte distances according to first pivot
    ℓ = min.(distance.(dofs, Ref(dofs[P[1]])), ℓ0)

    # compute upper bound of diameter of domain
    diam = 2 * maximum(ℓ) 

    f = Family(dofs)
    # Setting the descendants of the first node to be all dofs, ordering the distance 
    descendants(f)[P[1]] = sort([(ℓ[k], k) for k = 1 : length(ℓ)])
    parents(f) .= [(ℓ[k], P[1]) for k = 1 : length(ℓ)]

    # instantiate rack 
    rack = Rack(n_bins, r, diam, ℓ)

    # starting by looking in first bin
    first_bin = 1
    while ~iszero(first_bin)
        # obtain new pivot
        first_bin, pivot_id = next_dof(first_bin, rack)
        # if pivot was not in last bin

        if !iszero(pivot_id)
            new_pivot!(P, ℓ_out, rack, f, pivot_id, dofs, ℓ, checklist) 
            checklist[pivot_id] = false
        end
    end
    # Once all remaining dofs are in the smallest bin, we simply add them to the ordering in arbitrary order
    # First set the length-scale of the remaining dofs to the minimal resolution
    ℓ_out[setdiff(1 : length(dofs), P)] = ℓ[setdiff(1 : length(dofs), P)]
    P = unique(vcat(P, 1 : length(dofs)))
    return P, ℓ_out
end

function new_pivot!(P, ℓ, h::MutableBinaryMaxHeap, f::Family, pivot_id, pivot_ℓ, dofs) 
    # add pivot to ordering
    push!(P, pivot_id)
    pivot = dofs[pivot_id]
    pivot_parent = parents(f)[pivot_id]

    # remember the length scale for the output
    # ℓ[pivot_id] = pivot_ℓ
    # Iterating over descendants of the parent that are close enough to the parent to possibly be descendants of the pivot
        # Update the distance
    # @show length(close_descendants(pivot_parent[2], distance(dofs[pivot_parent[2]], pivot) + 2 * pivot_ℓ, f))
    for parent_descendant in close_descendants(pivot_parent[2], distance(dofs[pivot_parent[2]], pivot) + 2 * pivot_ℓ, f)
        distance_descendant_pivot = distance(dofs[parent_descendant[2]], pivot)
        if h.node_map[parent_descendant[2]] != 0
            # Update the distance
            h[parent_descendant[2]] = min(distance_descendant_pivot, h[parent_descendant[2]])
            ℓ[parent_descendant[2]] = min(distance_descendant_pivot, h[parent_descendant[2]])
            # Check if distance is small enough for descendant to be included as descendant of the pivot
            if distance_descendant_pivot < 2 * pivot_ℓ
                push!(descendants(f)[pivot_id], (distance_descendant_pivot, parent_descendant[2]))
            end
            # Check if pivot should become the descendant's next parent 
            if distance_descendant_pivot + 2 * ℓ[parent_descendant[2]] < pivot_ℓ && distance_descendant_pivot < parents(f)[parent_descendant[2]][1]
                parents(f)[parent_descendant[2]] = (distance_descendant_pivot, pivot_id)
            end
        end
    end 
    # Sorts the descendants of the new pivot by distance to the pivot
    sort!(descendants(f)[pivot_id], alg=QuickSort)
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
    # Initializing vector containing the reordering
    P = Vector{Int}(undef, 0)
    # choosing the first pivot as having minimal distance according to the "hard coded" distance
    push!(P, argmax(ℓ0))
    # updating the distances according to first pivot
    ℓ = min.(distance.(dofs, Ref(dofs[P[1]])), ℓ0) 
    f = Family(dofs)
    # Setting the descendants of the first node to be all dofs, ordering the distance 
    descendants(f)[P[1]] = sort([(ℓ[k], k) for k = 1 : length(ℓ)])
    parents(f) .= [(ℓ[k], P[1]) for k = 1 : length(ℓ)]

    # instantiate heap and remove the largest element (P[1])
    ℓ[P[1]] = Inf; h = MutableBinaryMaxHeap(ℓ); pop!(h); ℓ[P[1]] = min(ℓ0[P[1]], Inf)


    # starting by looking in first bin
    for k = 2 : length(ℓ)
        # obtain new pivot
        pivot_ℓ, pivot_id = top_with_handle(h); pop!(h)
        # if pivot was not in last bin
        new_pivot!(P, ℓ, h, f, pivot_id, pivot_ℓ, dofs) 
    end
    # Once all remaining dofs are in the smallest bin, we simply add them to the ordering in arbitrary order
    # First set the length-scale of the remaining dofs to the minimal resolution
    return P, ℓ
end
