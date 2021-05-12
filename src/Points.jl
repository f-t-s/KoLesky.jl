import StaticArrays.SVector
import LinearAlgebra.norm

# Points represent the geometric location of (groups of) measurements 
abstract type AbstractPoint end 

# A point that can be used to construct 
struct Point{d,RT}<:AbstractPoint
    location::SVector{d,RT}
end

function location(in::Point)
    return in.location
end

function Point(in::SVector{d,RT}) where {d,RT}
    return Point{d,RT}(in)
end

# Takes in a matrix and returns a Vector of Points with its columns as locations
function matrix2points(in::AbstractMatrix, dims::Int=1) 
    if !(dims==1 || dims==2)
        ArgumentError("dims should be 1 or two")
    elseif dims == 2
        in = in'
    end
    d, N = size(in)
    out = Vector{Point{d,eltype(in)}}(undef, N)
    for k = 1 : N
        out[k] = Point(SVector{d}(in[:, k]))
    end
    return out
end

# Function that computes the squared Euclidean distance between two poins
function squared_euclidean_distance(p::Point{d,RT}, q::Point{d,RT}) where {d,RT<:Real}
    diff = location(p) - location(q)
    return sum(diff.^2)
end

# Function that computes the squared Euclidean distance between two poins
function euclidean_distance(p::Point{d,RT}, q::Point{d,RT}) where {d,RT<:Real}
    return norm(location(p) - location(q))
end