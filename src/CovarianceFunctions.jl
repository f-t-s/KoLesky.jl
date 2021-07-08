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



# The Matern covariance function
struct MaternCovariance5_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Matern covariance function
function (cov::MaternCovariance5_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return (1+sqrt(5)*dist/sigma + 5*dist^2/(3*sigma^2)) * exp(-sqrt(5)*dist/sigma)
end

function (cov::MaternCovariance5_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    D2F(t,a) = -(2*a^2+2*sqrt(5)*a*t-5*t^2)/(3*a^4) * exp(-sqrt(5)*t/a);
    D4F(t,a) = 25*(8*a^2-7*sqrt(5)*a*t+5*t^2)/(3*a^6) * exp(-sqrt(5)*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*(1+sqrt(5)*dist/sigma + 5*dist^2/(3*sigma^2)) * exp(-sqrt(5)*dist/sigma)
end


# function (cov::ExponentialCovariance)(x::ΔδPointMeasurement, y::PointMeasurement)
#     return cov(x, ΔδPointMeasurement(y))
# end

# function (cov::ExponentialCovariance)(x::PointMeasurement, y::ΔδPointMeasurement)
#     return (cov::ExponentialCovariance)(y,x)
# end

# The exponential covariance function
struct GaussianCovariance{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

function (cov::GaussianCovariance)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return exp(-dist^2/(2*sigma^2))
end

function (cov::GaussianCovariance)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    D2F(t,rho) = (t - 2*rho^2)/(rho^4)*exp(-t/(2*rho^2));
    D4F(t,rho) = (8*rho^4-8*rho^2*t+t^2)/(rho^8)*exp(-t/(2*rho^2));
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist^2,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist^2,sigma) + w2_x*w2_y*exp(-dist^2/(2*sigma^2))
end


function (cov::AbstractCovarianceFunction)(x::ΔδPointMeasurement, y::PointMeasurement)
    return cov(x, ΔδPointMeasurement(y))
end

function (cov::AbstractCovarianceFunction)(x::PointMeasurement, y::ΔδPointMeasurement)
    return (cov::AbstractCovarianceFunction)(y,x)
end