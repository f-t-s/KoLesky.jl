import StaticArrays.SVector

abstract type AbstractMeasurement end
abstract type AbstractPointMeasurement{d}<:AbstractMeasurement end

struct PointMeasurement{d}<:AbstractPointMeasurement{d}
    coord::SVector{Float64,d}
end 