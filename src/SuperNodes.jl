include("DOF.jl")

struct SuperNode{Ti}
    parents::AbstractVector{Ti}
    children::AbstractVector{Ti}
end

# Set of disjoint super nodes
struct SuperNodalPartition{Ti}
    nodes::Vector{SuperNode{Ti}} 
    lookup::Vector{Ti}
end

function SuperNodalPartition(N::Ti) where Ti
    return SuperNodalPartition(Vector{SuperNode{Ti}}(undef, 0), zeros(Ti, N))
end

function SuperNodalPartition{Ti}(N) where Ti
    return SuperNodalPartition(Vector{SuperNode{Ti}}(undef, 0), zeros(Ti, N))
end

import Base.push!
function push!(sns::SuperNodalPartition{Ti}, items...) where Ti
    # store old number of supernodes
    old_length = length(sns.nodes)
    push!(sns.nodes, items...)
    # let lookup entries corresponding to parents in item point to old_length + item_index 
    for (item_index, item) ∈ enumerate(items)
        for id ∈ item.parents
            # Ensures that the parent sets of the supernodes are disjoint
            @assert sns.lookup[id] == zero(Ti)
            sns.lookup[id] = old_length + item_index
        end
    end
end

function append!(sns::SuperNodalPartition{Ti}, items) where Ti
    # store old number of supernodes
    old_length = length(sns.nodes)
    push!(sns.nodes, items...)
    # let lookup entries corresponding to parents in item point to old_length + item_index 
    for (item_index, item) ∈ enumerate(items)
        for id ∈ item.parents
            # Ensures that the parent sets of the supernodes are disjoint
            @assert sns.lookup[id] == zero(Ti)
            sns.lookup[id] = old_length + item_index
        end
    end
end


import Base.union
function union(𝒩::SuperNodalPartition{Ti}, ℳ::SuperNodalPartition{Ti}) where Ti
    out = SuperNodalPartition{Ti}(length(𝒩.lookup))
    append!(out, ℳ.nodes)
    append!(out, 𝒩.nodes)
    return out
end

function is_complete(𝒩::SuperNodalPartition)
    return ~any(𝒩.lookup .== 0)
end


function containing_supernode(𝒩, i)
    return 𝒩.nodes[𝒩.lookup[i]]
end

function refine(𝒩::SuperNodalPartition{Ti}, P, ℓ, r, ρ, λ, dofs) where Ti
    parent_buffer = similar(P)
    children_buffer = similar(P)

    # α is chosen such that α * ℓ ≥ (1 + α λ⁻¹) * ℓ 
    # This ensures the children sets of nodes picked in the future, are contained among the children of the supernode that they are a parent in
    α = 1 / (1 - λ^(-1))

    # Create the refined supernodal decomposition
    ℳ = SuperNodalPartition{Ti}(length(P))
    checklist = trues(length(P))
    for  i = 1 : length(P)
        if checklist[i]
            σ = containing_supernode(𝒩, i)
            parent_offset = 0
            children_offset = 0
            for j in σ.children
                dist = distance(dofs[i], dofs[j])
                if dist ≤ 2 * α * ρ * λ * r 
                    # add index as child node
                    children_offset += 1
                    children_buffer[children_offset] = j 
                    if  dist ≤ ρ * α * λ * r && checklist[j]
                        # Add index to the parent list of supernode 
                        parent_offset += 1
                        parent_buffer[parent_offset] = j
                        checklist[j] = false 
                    end
                end
            end
            # add new supernode 
            push!(ℳ, SuperNode(parent_buffer[1:parent_offset], children_buffer[1:children_offset]))
        end
    end

    𝒩_geq = SuperNodalPartition{Ti}(length(P))
    for σ ∈ 𝒩.nodes
        parent_offset = 0
        children_offset = 0
        for i ∈ σ.parents
            if r < ℓ[i] ≤ r * λ
                parent_offset += 1
                parent_buffer[parent_offset] = i
            end
        end
        for i ∈ σ.children
            if r < ℓ[i] ≤ r * λ
                children_offset += 1
                children_buffer[children_offset] = i
            end
        end
        push!(𝒩_geq, SuperNode(parent_buffer[1:parent_offset], children_buffer[1:children_offset]))
    end
    return ℳ, 𝒩_geq
end

function reduce(ρ, P, ℓ, S̄::SuperNodalPartition{Ti}, dofs) where Ti
    S̃ = SuperNodalPartition{Ti}(length(P))
    # we will be using the buffer by marking nodes that are present as $1$.
    children_buffer = similar(P)
    children_buffer .= zero(eltype(children_buffer))
    revP = similar(P)
    revP[P] = 1 : length(P)
    for σ ∈ S̄
        for (i, j) ∈ Iterators.product(σ.parents, σ.children)
            # ordering reversed because P denotes maxmin and not reverse maxmin ordering
            if distance(dofs[i], dofs[j]) ≤ ρ * ℓ[i] && revP[i] ≥ revP[j]
                # Add j to the children that will be added
                children_buffer[j] = 1
            end
        end
        push!(S̃, SuperNode{Ti}(copy(σ.parents), findall(isone, children_buffer)))
    end
    return S̃
end

function partition_into_supernodes(dofs, ℓ, P, ρ, λ)
    Ti = eltype(P)
#     parent_buffer = similar(P)
#     children_buffer = similar(P)
    𝒩 = SuperNodalPartition{Ti}(length(P))
    push!(𝒩, SuperNode(copy(P), copy(P)))

    # Initializing the partition into supernodes that only contain nodes in a given range of length scales.
    𝒩_geq = SuperNodalPartition{Ti}(length(P))
    push!(𝒩_geq, SuperNode(P[1:1], P[1:1]))

    # threshhold of the scale being considered in a given iteration
    r = ℓ[P[2]] / λ

    while r > ℓ[P[end]] / λ
        @show 𝒩_geq.lookup
        𝒩, 𝒩_geq_new = refine(𝒩, P, ℓ, r, ρ, λ, dofs) 

        @show 𝒩_geq_new.lookup
        𝒩_geq = union(𝒩_geq, 𝒩_geq_new)
        r /= λ 
    end
    @show 𝒩_geq.lookup
    return 𝒩_geq
end