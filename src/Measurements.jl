import StaticArrays.SVector
import LinearAlgebra.norm

abstract type AbstractMeasurement end
abstract type AbstractPointMeasurement{d}<:AbstractMeasurement end

function get_coordinate(m::AbstractPointMeasurement)
    return m.coordinate
end
struct PointMeasurement{d}<:AbstractPointMeasurement{d}
    coordinate::SVector{d,Float64}
end 

struct ΔδPointMeasurement{Tv,d}<:AbstractPointMeasurement{d}
    coordinate::SVector{d,Float64}
    weight_Δ::Tv
    weight_δ::Tv
end

function ΔδPointMeasurement(in::PointMeasurement{d}) where d
    return ΔδPointMeasurement{Float64,d}(in.coordinate, zero(Float64), one(Float64))
end

function point_measurements(x::Matrix; dims=1)
    if dims == 2  
        x = x'
    elseif dims !=1
        error("keyword argumend \"dims\" should be 1 or 2")
    end
    d = size(x, 1)
    return [PointMeasurement{d}(SVector{d,Float64}(x[:, k])) for k = 1 : size(x, 2)]
end

