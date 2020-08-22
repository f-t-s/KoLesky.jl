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