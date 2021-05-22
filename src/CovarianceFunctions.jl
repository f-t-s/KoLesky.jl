import LinearAlgebra.norm

abstract type AbstractCovarianceFunction end 

# automatically implements a mutating batched version of a given covariance function 
function (cov<:AbstractCovarianceFunction)(out::AbstractMatrix, x_vec::AbstractVector{<:AbstractMeasurement}, y_vec{<:AbstractMeasurement})
    for cartesian in CartesianIndices(out) 
        out[cartesian] = cov(x_vec[cartesian[1]], y_vec[cartesian[2]])
    end
end

# The exponential covariance function
struct ExponentialCovariance{Tv}
    length_scale::Tv
end

function ExponentialCovariance(length_scale)
    return ExponentialCovariance{Tv}(Tv)
end

# Exponential covariance function
function (cov::ExponentialCovariance)(x::PointMeasurement, y::PointMeasurement)
    return exp(-norm(x.coordinate - y.coordinate) / cov.length_scale)
end

# TODO: Implement Δδ
function (cov::ExponentialCovariance)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
end

function (cov::ExponentialCovariance)(x::ΔδPointMeasurement, y::PointMeasurement)
    return cov(x, ΔδPointMeasurement(y))
end