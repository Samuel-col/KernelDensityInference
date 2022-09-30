# Funciones base de inferenciaDensidadKernel
using Statistics, StatsBase, LinearAlgebra

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
    new_x_ind = sample(1:(nx+ny),nx)
    new_x = U[new_x_ind]
    new_y = symdiff(U,new_x)

    return new_x,new_y
end

## Reordenar muestra
function shuffle_sample(x::Vector{T})::Vector{T} where T<:Real
    n = length(x)
    #new_ids = sample(1:n,n,replace = false)
    #return x[new_ids]
    new_x = sample(x,n,replace = false)
    return new_x
end
