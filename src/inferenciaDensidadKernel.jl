# Pruebas distribucionales basadas en la densidad kernel
module KernelTests

    # Funciones a exportar
    export KernelDensity, BivariateKernelDensity
    export density, discretize, plot, plot!
    export sameDistributionTest, independencyTest

    # Paquetes
    using Statistics, StatsBase, LinearAlgebra, Plots
    import Base: display
    import Plots: plot, plot!, heatmap, heatmap!
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

    abstract type AbstractKernelDensity <: Any end

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
        y = density(x,KD)
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

    abstract type AbstractHypothesysTest <: Any end

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

    struct BivariateKernelDensity <: AbstractKernelDensity
        n::Int
        h::Tuple{Float64,Float64}
        domain::Tuple{Tuple{T,T},Tuple{T,T}} where T<:Real
        # xi::Tuple{Vector{T},Vector{T}} where T<:Real
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

    ## Método para graficar BivariateKernelDensity
    function plot(BKD::BivariateKernelDensity;nGrid::Int  = 160,args...)::Plots.Plot

        x_grid = range(BKD.domain[1]...,length = nGrid)
        y_grid = range(BKD.domain[2]...,length = nGrid)
        dty = [density((x,y),BKD) for y in y_grid, x in x_grid]
        
        heatmap(x_grid,y_grid,dty,
            color = :coolwarm;
            args...
        )
    end

    function plot!(BKD::BivariateKernelDensity;nGrid::Int  = 160,args...)::Plots.Plot
        x_grid = range(BKD.domain[1]...,length = nGrid)
        y_grid = range(BKD.domain[2]...,length = nGrid)
        dty = [density((x,y),BKD) for y in y_grid, x in x_grid]
        
        heatmap!(x_grid,y_grid,dty,
            color = :coolwarm;
            args...
        )
    end


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
end