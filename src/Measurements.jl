import StaticArrays.SVector
import LinearAlgebra.norm

abstract type AbstractMeasurement end
abstract type AbstractPointMeasurement{d}<:AbstractMeasurement end

struct PointMeasurement{d}<:AbstractPointMeasurement{d}
    coordinate#::SVector{Float64,d}
end 

struct ΔδPointMeasurement{Tv,d}<:AbstractPointMeasurement{d}
    coordinate::SVector{Float64,d}
    weight_Δ::Tv
    weight_δ::Tv
end

function ΔδPointMeasurement(in::PointMeasurement{d}) where d
    return ΔδPointMeasurement{Float64,d}(in.coordinate, zero(Float64), one(Float64))
end
