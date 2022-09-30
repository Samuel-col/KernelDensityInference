# Estimadores de densidad kernel para inferenciaDensidadKernel
import Base: display
import Plots: plot, plot!, heatmap, heatmap!
using Plots

## Ancho de banda óptimo
function optimum_bandwidth(x::Vector{T})::Float64 where T<:Real
    n = length(x)
    sig = std_hat(x)
    return sig*(4/(3n))^(1/5) # Revisar la teoría de fondo. Puntualmente R(f'') con f la pdf de una normal μ,σ²
end

abstract type AbstractKernelDensity <: Any end


## -------------------- Kernel univariado --------------------------------

## Objeto KernelDensity
struct KernelDensity <: AbstractKernelDensity
    n::Int
    h::Float64
    domain::Tuple{T,T} where T<:Real
    pts::Vector{T} where T<:Real

    ## Métido constructor
    function KernelDensity(x::Vector{T};h::Union{T,Missing} = missing) where T<:Real
        n = length(x)
        if ismissing(h)
            h = optimum_bandwidth(x)
        end
        extr = extrema(x)
        range = extr[2] - extr[1]
        domain = extr .+ (-range*0.3,range*0.3)

        new(n,h,domain,x)
    end

end

## Evaluar la densidad kernel en un punto
function density(x::T,KD::KernelDensity) where T<:Real

    return mean([K(x-p,h = KD.h) for p in KD.pts])
end

## Evaluar la densidad kernel en varios puntos
function density(x::Vector{T},KD::KernelDensity) where T<:Real
    
    return [density(xi,KD) for xi in x]
end

## Descripción de KernelDensity
function display(KD::KernelDensity)
    report = """
        KernelDensity:
            n = $(KD.n)
            Bandwidth (h) = $(KD.h)
            domain = ($(KD.domain[1]), $(KD.domain[2]))
    """
    println(report)
end

## Discretizar la función de densidad kernel evaluándola en una grilla
function discretize(KD::KernelDensity;nGrid::Int = 400)::Tuple{Vector{Float64},Vector{Float64}}
    a = KD.domain[1]
    b = KD.domain[2]
    Δ = (b-a)/nGrid
    x = collect(a:Δ:b)
    y = density(x,KD)
    return x,y
end


## -------------------- Kernel bivariado --------------------------------

struct BivariateKernelDensity <: AbstractKernelDensity
    n::Int
    h::Tuple{Float64,Float64}
    domain::Tuple{Tuple{T,T},Tuple{T,T}} where T<:Real
    pts::Vector{Tuple{T,T}} where T<:Real

    ## Métido constructor
    function BivariateKernelDensity(x::Vector{T},y::Vector{T};
                    hx::Union{T,Missing} = missing,
                    hy::Union{T,Missing} = missing) where T<:Real
        
        if length(x) != length(y)
            stop("x and y must have the same length.")
        end

        n = length(x)
        
        if ismissing(hx)
            hx = optimum_bandwidth(x)
        end
        if ismissing(hy)
            hy = optimum_bandwidth(y)
        end
        
        extr = extrema(x)
        range = extr[2] - extr[1]
        domainx = extr .+ (-range*0.3,range*0.3)

        extr = extrema(y)
        range = extr[2] - extr[1]
        domainy = extr .+ (-range*0.3,range*0.3)

        pts = [(x[i],y[i]) for i in 1:n]

        new(n,(hx,hy),(domainx,domainy),pts)
    end
end

## Evaluar la densidad kernel bivariado en un punto
function density(x::Tuple{T,T},BKD::BivariateKernelDensity) where T<:Real

    return mean([K(x[1]-p[1],h = BKD.h[1])*K(x[2]-p[2],h = BKD.h[2]) for p in BKD.pts])
end

## Evaluar la densidad kernel bivariado en varios puntos
function density(x::Vector{Tuple{T,T}},BKD::BivariateKernelDensity) where T<:Real
    
    return [density(xi,BKD) for xi in x]
end

## Descripción de BivariateKernelDensity
function display(BKD::BivariateKernelDensity)
    report = """
        KernelDensity:
            n = $(BKD.n)
            Bandwidths (hx,hy) = $(BKD.h)
            domainx = $(BKD.domain[1])
            domainy = $(BKD.domain[2])
    """
    println(report)
end

## Discretizar la función de densidad kernel bivariado evaluándola en una grilla
function discretize(BKD::BivariateKernelDensity;nGrid::Int = 400)
    x_grid = range(BKD.domain[1]...,length = nGrid)
    y_grid = range(BKD.domain[2]...,length = nGrid)
    pts = [(x,y) for y in y_grid, x in x_grid] |> vec
    d_pts = density(pts,BKD)
    return pts,d_pts
end

