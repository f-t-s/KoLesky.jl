import LinearAlgebra.norm

abstract type AbstractCovarianceFunction{Tv} end 

# automatically implements a mutating batched version of a given covariance function 
function (cov::AbstractCovarianceFunction{Tv})(out::AbstractMatrix{Tv}, x_vec::AbstractVector{<:AbstractMeasurement}, y_vec::AbstractVector{<:AbstractMeasurement}) where Tv
    for cartesian in CartesianIndices(out) 
        out[cartesian] = cov(x_vec[cartesian[1]], y_vec[cartesian[2]])
    end
end

# automatically implements a mutating batched version of a given covariance function, using symmetry
function (cov::AbstractCovarianceFunction{Tv})(out::AbstractMatrix{Tv}, x_vec::AbstractVector{<:AbstractMeasurement}) where Tv
    for cartesian in CartesianIndices(out) 
        if cartesian[1] >= cartesian[2]
            out[cartesian] = cov(x_vec[cartesian[1]], x_vec[cartesian[2]])
        else
            out[cartesian] = out[cartesian[2], cartesian[1]]
        end
    end
end


# The exponential covariance function
struct ExponentialCovariance{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Exponential covariance function
function (cov::ExponentialCovariance)(x::PointMeasurement, y::PointMeasurement)
    return exp(-norm(x.coordinate - y.coordinate) / cov.length_scale)
end

# TODO: Implement Δδ
function (cov::ExponentialCovariance)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    D2F(t,a) = (-(a^2+a*t-t^2)/(3*a^4)-(a+t)/(3*a^3)) * exp(-t/a);
    D4F(t,a) = ((4*a-t)/(3*a^5) + (4*a^2-6*a*t+t^2)/(3*a^6)) * exp(-t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*exp(-dist/sigma)
end


function (cov::ExponentialCovariance)(x::ΔδPointMeasurement, y::PointMeasurement)
    return cov(x, ΔδPointMeasurement(y))
end

function (cov::ExponentialCovariance)(x::PointMeasurement, y::ΔδPointMeasurement)
    return (cov::ExponentialCovariance)(y,x)
end
