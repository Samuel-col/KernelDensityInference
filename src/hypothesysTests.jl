# Pruebas de hipótesis para inferenciaDensidadKernel
import QuadGK: quadgk

abstract type AbstractHypothesysTest <: Any end


## -------------------- Igualdad de distribuciones --------------------------------

## Estadística de Igualdad de distribuciones
function sameDistributionTest_statistic(x::Vector{T},y::Vector{T})::Float64 where T<:Real
    kd_x = KernelDensity(x)
    kd_y = KernelDensity(y)

    a = min(kd_x.domain[1],kd_y.domain[1])
    b = max(kd_x.domain[2],kd_y.domain[2])
    
    Tc, = quadgk(x -> ( density(x,kd_x) - density(x,kd_y) )^2, a, b,
        maxevals = 10e3,
        atol = 10e-5
    )
    
    return Tc
end

## Igualdad de distribuciones
struct sameDistributionTest <: AbstractHypothesysTest
    n1::Int
    n2::Int
    Tc::Float64
    pValue::Float64
    NIter::Int
    plot::Union{Plots.Plot,Missing}

    function sameDistributionTest(x::Vector{T},y::Vector{T}; 
                                    NIter::Int = 500,plt::Bool = true) where T<: Real
        
        nx = length(x)
        ny = length(y)
        Tc = sameDistributionTest_statistic(x,y)

        T_distribution = []
        for i in 1:NIter
            xi, yi = mix_samples(x,y)
            Ti = sameDistributionTest_statistic(xi,yi)
            push!(T_distribution,Ti)
        end

        pValue = mean(Tc .< T_distribution)

        if plt
            T_distribution = convert(Vector{Float64},T_distribution)
            kd_T = KernelDensity(T_distribution)
            plot = Plots.plot(kd_T,
                        title = "T-statistic distribution",
                        xlabel = "t",
                        ylabel = "Density",
                        legend = false
                        )
            plot = Plots.vline!(plot,[Tc],linestyle = :dash)
        else
            plot = missing
        end

        new(nx,ny,Tc,pValue,NIter,plot)
    end

end

## Descripción de sameDistributionTest
function display(test::sameDistributionTest)
    result = """
    Equal distributions test:
        Samples length: 
            n1 = $(test.n1)
            n2 = $(test.n2)
        Null Hypothesys:
            Both samples have the same probability density function.
        Test Statistic:
            Tc = $(test.Tc)
            p-value = $(test.pValue)
                (aproximated using $(test.NIter) iterations).
    """
    println(result)
end


## -------------------- Indenpendencia de poblaciones --------------------------------


## Estadística de Independencia de variables
function independencyTest_statistic(x::Vector{T},y::Vector{T}) where T<:Real
    kd_x = KernelDensity(x)
    kd_y = KernelDensity(y)
    kd_xy = BivariateKernelDensity(x,y)

    Tc = (density(kd_xy.pts,kd_xy) ./ (density(x,kd_x) .* density(y,kd_y)) ) .|> log |> mean

    return Tc

end

## Independencia de variables
struct independencyTest <: AbstractHypothesysTest
    n::Int
    Tc::Float64
    pValue::Float64
    NIter::Int
    plot::Union{Plots.Plot,Missing}

    function independencyTest(x::Vector{T},y::Vector{T};
                                NIter::Int = 500, plt::Bool = true) where T<:Real
        nx = length(x)
        ny = length(y)

        if nx != ny
            stop("x and y must have the same length.")
        end

        Tc = independencyTest_statistic(x,y)
        

        T_distribution = []
        for i in 1:NIter
            y_star = shuffle_sample(y)
            Ti = independencyTest_statistic(x,y_star)
            push!(T_distribution,Ti)
        end

        pValue = mean(Tc .< T_distribution)

        if plt
            T_distribution = convert(Vector{Float64},T_distribution)
            kd_T = KernelDensity(T_distribution)
            plot = Plots.plot(kd_T,
                        title = "T-statistic distribution",
                        xlabel = "t",
                        ylabel = "Density",
                        legend = false
                        )
            plot = Plots.vline!(plot,[Tc],linestyle = :dash)
        else
            plot = missing
        end

        new(nx,Tc,pValue,NIter,plot)
    end

end

## Descripción de independencyTest
function display(test::independencyTest)
    result = """
    Independency test:
        Sample length: 
            n = $(test.n)
        Null Hypothesys:
            Variables x and y are independent.
        Test Statistic:
            Tc = $(test.Tc)
            p-value = $(test.pValue)
                (aproximated using $(test.NIter) iterations).
    """
    println(result)
end



## Graficar resultado de una prueba de hipótesis
function plot(test::AbstractHypothesysTest)
    return test.plot
end