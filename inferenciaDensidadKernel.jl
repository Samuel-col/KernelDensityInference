# Pruebas distribucionales basadas en la densidad kernel
module KernelTests

    # Funciones a exportar
    export KernelDensity, density, discretize, plot, sameDistributionTest

    # Paquetes
    using Statistics, StatsBase, LinearAlgebra, Plots
    import Base: display
    import Plots: plot
    import QuadGK: quadgk

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

    ## Estimación de la desviación estándar
    function std_hat(x::Vector{T})::Float64 where T<:Real
        QQ = quantile(x,[0.25,0.75])
        RIC = QQ[2] - QQ[1]
        return min(RIC/1.35, std(x))
    end

    ## Ancho de banda óptimo
    function optimum_bandwidth(x::Vector{T})::Float64 where T<:Real
        n = length(x)
        sig = std_hat(x)
        return sig*(4/(3n))^(1/5) # Revisar la teoría de fondo. Puntualmente R(f'') con f la pdf de una normal μ,σ²
    end

    ## Objeto KernelDensity
    struct KernelDensity <: Any
        n::Int
        h::Float64
        domain::Tuple{T,T} where T<:Real
        xi::Vector{T} where T<:Real

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

        return mean([K(x-xi,h = KD.h) for xi in KD.xi])
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

    ## Método para graficar KernelDensity
    function plot(KD::KernelDensity;args...)::Plots.Plot
        plot(x -> density(x,KD), KD.domain[1],KD.domain[2];args...)
    end

    function plot!(KD::KernelDensity;args...)::Plots.Plot
        plot!(x -> density(x,KD), KD.domain[1],KD.domain[2];args...)
    end

    ## Discretizar la función de densidad kernel evaluándola en una grilla
    function discretize(KD::KernelDensity;nGrid::Int = 400)::Tuple{Vector{Float64},Vector{Float64}}
        a = KD.domain[1]
        b = KD.domain[2]
        Δ = (b-a)/nGrid
        x = collect(a:Δ:b)
        y = [density(xi,KD) for xi in x]
        return x,y
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
        new_ids = sample(1:n,n,replace = false)
        return x[new_ids]
    end


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
    struct sameDistributionTest 
        n1::Int
        n2::Int
        Tc::Float64
        pValue::Float64
        NIter::Int
        plot::Union{Plots.Plot,Missing}

        function sameDistributionTest(x::Vector{T},y::Vector{T}; 
                                        NIter::Int = 200,plt::Bool = true) where T<: Real
            
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

end