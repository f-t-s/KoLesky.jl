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

struct MaternCovariance7_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Matern covariance function
function (cov::MaternCovariance7_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (15*a^3+15*sqrt(7)*a^2*t+42*a*t^2+7*sqrt(7)*t^3)/(15*a^3)*exp(-sqrt(7)*t/a);
    return F(dist,sigma)
end

function (cov::MaternCovariance7_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = (15*a^3+15*sqrt(7)*a^2*t+42*a*t^2+7*sqrt(7)*t^3)/(15*a^3)*exp(-sqrt(7)*t/a);
    D2F(t,a) = -7*(6*a^3+6*sqrt(7)*a^2*t+7*a*t^2-7*sqrt(7)*t^3)/(15*a^5)*exp(-sqrt(7)*t/a);
    D4F(t,a) = 49*(8*a^3+8*sqrt(7)*a^2*t-56*a*t^2+7*sqrt(7)*t^3)/(15*a^7)*exp(-sqrt(7)*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

struct MaternCovariance9_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Matern covariance function
function (cov::MaternCovariance9_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (35*a^4+105*a^3*t+135*a^2*t^2+90*a*t^3+27*t^4)/(35*a^4)*exp(-3*t/a);
    return F(dist,sigma)
end

function (cov::MaternCovariance9_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = (35*a^4+105*a^3*t+135*a^2*t^2+90*a*t^3+27*t^4)/(35*a^4)*exp(-3*t/a);
    D2F(t,a) = -9*(10*a^4+30*a^3*t+27*a^2*t^2-9*a*t^3-27*t^4)/(35*a^6)*exp(-3*t/a);
    D4F(t,a) = 81*(8*a^4+24*a^3*t-72*a*t^3+27*t^4)/(35*a^8)*exp(-3*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

struct MaternCovariance11_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Matern covariance function
function (cov::MaternCovariance11_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (945*a^5+945*sqrt(11)*a^4*t+4620*a^3*t^2+1155*sqrt(11)*a^2*t^3+1815*a*t^4+121*sqrt(11)*t^5)/(945*a^5)*exp(-sqrt(11)*t/a);
    return F(dist,sigma)
end

function (cov::MaternCovariance11_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = (945*a^5+945*sqrt(11)*a^4*t+4620*a^3*t^2+1155*sqrt(11)*a^2*t^3+1815*a*t^4+121*sqrt(11)*t^5)/(945*a^5)*exp(-sqrt(11)*t/a);
    D2F(t,a) = -11*(210*a^5+210*sqrt(11)*a^4*t+825*a^3*t^2+55*sqrt(11)*a^2*t^3-484*a*t^4-121*sqrt(11)*t^5)/(945*a^7)*exp(-sqrt(11)*t/a);
    D4F(t,a) = 121*(120*a^5+120*sqrt(11)*a^4*t+264*a^3*t^2-176*sqrt(11)*a^2*t^3-847*a*t^4+121*sqrt(11)*t^5)/(945*a^9)*exp(-sqrt(11)*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end


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