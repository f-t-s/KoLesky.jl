import LinearAlgebra.norm

# Exponential covariance function
function exponential(x::PointMeasurement, y::PointMeasurement; length_scale)
    return exp(-norm(x.coordinate - y.coordinate) / length_scale)
end

# TODO: Implement Δδ
function exponential(x::ΔδPointMeasurement, y::ΔδPointMeasurement; length_scale)
end

function exponential(x::ΔδPointMeasurement, y::PointMeasurement; length_scale)
    return exponential(x, ΔδPointMeasurement(y); length_scale)
end

function exponential(x::AbstractVector{<:AbstractMeasurement}, y::AbstractVector{<:AbstractMeasurement}; length_scale)
    out = Matrix{Float64}(undef, length(x), length(y))
    for (i, xval) in enumerate(x)
        for (j, yval) in enumerate(y)
            out[i, j] = exponential(x, y; length_scale)
        end
    end
    return out
end 

function exponential(x::AbstractVector{PointMeasurement{d}}, y::AbstractVector{PointMeasurement{d}}; length_scale)
    out = Matrix{Float64}(undef, length(x), length(y))
    for (i, xval) in enumerate(x)
        for (j, yval) in enumerate(y)
            out[i, j] = norm(xval - yval) / length_scale
        end
    end
    out .= exp.(-out)
end
