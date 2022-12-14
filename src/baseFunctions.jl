# Funciones base de inferenciaDensidadKernel
using Statistics, StatsBase, LinearAlgebra, Base.Threads

## Densidad normal
function gauss_pdf(x::T; mu::T = 0, sig::T = 1)::Float64 where T<:Real
    ker = exp(-1/2*((x-mu)/sig)^2)
    norm = (sqrt(2π)*sig)
    return ker/norm
end

## Kernel gaussiano
function K(x::T;h::T = 1)::Float64 where T<:Real
    return gauss_pdf(x, mu = 0.0, sig = h)
end

## Estimación robusta de la desviación estándar
function std_hat(x::Vector{T})::Float64 where T<:Real
    QQ = quantile(x,[0.25,0.75])
    RIC = QQ[2] - QQ[1]
    return min(RIC/1.35, std(x))
end

## Remuestrear a partir de dos muestras
function mix_samples(x::Vector{T},y::Vector{T})::Tuple{Vector{T},Vector{T}} where T<:Real
    nx = length(x)
    ny = length(y)
    U = copy(x)
    U = append!(U,y)
    new_x_ind = sample(1:(nx+ny),nx,replace = false)
    new_x = U[new_x_ind]
    new_y_ind = symdiff(1:(nx+ny),new_x_ind)
    new_y = U[new_y_ind]

    return new_x,new_y
end

# function mix_samples!(samps::Vector{Vector{T}}) where T<:Real # ::Tuple{Vector{T},Vector{T}}
#     N = length.(samps)
#     nx, ny = N[1], N[2]
#     U = append!(copy.(samps)...)
#     new_x_ind = sample(1:(nx+ny),nx,replace = false)
#     samps[1] = U[new_x_ind]
#     new_y_ind = symdiff(1:(nx+ny),new_x_ind)
#     samps[2] = U[new_y_ind]
#     nothing
#     #return new_x,new_y
# end

## Reordenar muestra
function shuffle_sample(x::Vector{T})::Vector{T} where T<:Real
    n = length(x)
    #new_ids = sample(1:n,n,replace = false)
    #return x[new_ids]
    new_x = sample(x,n,replace = false)
    return new_x
end

## Hacer validación cruzada
function crossValidation_kr(f::Union{Function,DataType},x::Vector{T},
                            y::Vector{T};args...)::Vector{Any} where T <: Real
    n = length(x)
    indices = collect(1:n)
    results = Vector{Any}(missing,n)
    @threads for k in indices
        new_indices = symdiff(indices,[k])
        results[k] = f(x[new_indices],y[new_indices];args...)
    end
    return results
end

## Linear Regression
function linearRegression(x::Vector{T},y::Vector{T}) where T <: Real
    
    n = length(x)
    if length(y) != n
        stop("x and y length must be the same.")
    end

    X = hcat(ones(n),x)

    β = inv(X'X)*(X'y)

    ε = y .- X*β

    return Dict("beta" => β, "residuals" => ε)
end
