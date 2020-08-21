import StaticArrays.SVector
import StaticArrays.norm

abstract type AbstractDOF end

struct Point{d, T}<:AbstractDOF
    pos::SVector{d,T}
    id::Int
end

function distance(p1::Point{d,T}, p2::Point{d,T}) where {d, T}
    # return norm(p1.pos - p2.pos)
    out = zero(T)
    @fastmath @inbounds @simd for k = 1 : d
        out += (p1.pos[k] - p2.pos[k])^2
    end
    return sqrt(out)
end

# function that takes in a matrix and returns a n array of points corresponding to dim-slices of the matrix.
function mat2points(x::AbstractMatrix; dims::Int=1)
    @assert dims ∈ [1,2]
    if dims == 2 
        x = x'
    end

    points = Vector{Point{size(x, 1), eltype(x)}}(undef, size(x, 2))
    for k = 1 : size(x, 2) 
        points[k] = Point{size(x, 1), eltype(x)}(SVector{size(x, 1), eltype(x)}(x[:, k]), k)
    end

    return points
end




