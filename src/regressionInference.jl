# Inferencia para regresión kernel

struct noEffectTest <: AbstractHypothesysTest

    n::Int
    Tc::Float64
    pValue::Float64
    NIter::Int
    plot::Union{Plots.Plot,Missing}

    function noEffectTest(x::Vector{T},y::Vector{T};
                            NIter::Int = 500, plt::Bool = true) where T <: Real
        
        kr = KernelRegression(x,y)
        Tc = Fstat(kr)

        T_distribution = zeros(NIter)
        @threads for i = 1:NIter
            y_star = shuffle_sample(y)
            kr_star = KernelRegression(x,y_star)
            T_distribution[i] = Fstat(kr_star)
        end

        pValue = mean(Tc .< T_distribution)

        if plt
            T_distribution = convert(Vector{Float64},T_distribution)
            kd_T = KernelDensity(T_distribution)
            plot = Plots.plot(kd_T,
                        title = "T-statistic distribution",
                        xlabel = "t",
                        ylabel = "Density",
                        legend = false,
                        xlim = (0,max(Tc,maximum(T_distribution))*1.2)
                        )
            plot = Plots.vline!(plot,[Tc],linestyle = :dash)
        else
            plot = missing
        end

        new(length(x),Tc,pValue,NIter,plot)
    end
end

## Descripción de noEffectTest
function display(test::noEffectTest)
    result = """
        No Effect Test:
            Sample length: 
                n = $(test.n)
        Null Hypothesys:
            x has no effect on y.
        Test Statistic:
            Tc = $(test.Tc)
            p-value = $(test.pValue)
                (aproximated using $(test.NIter) iterations).
    """
    println(result)
end


## No Effect Graphics test
function HoVar(kr::KernelRegression,x::T;σ2::T = 1.0)::Float64 where T<:Real
    w = [K(x - kr.x[j],h = kr.h) for j in 1:(kr.n)]
    w = w./sum(w)
    s = sum((w .- 1/kr.n).^2)
    return σ2 * s
end    

function HoVar(kr::KernelRegression,x::Vector{T})::Vector{Float64} where T<:Real
    σ2 = sig2(kr)
    return [HoVar(kr,xi,σ2 = σ2) for xi in x]        
end

function noEffectGraphicTest(x::Vector{T},y::Vector{T}; alpha::Float64 = 0.05, nGrid::Int = 160,args...) where T<:Real
    kr = KernelRegression(x,y)

    a,b = minimum(kr.x), maximum(kr.x)
    rango = b-a
    domain = (a-0.1rango,b+0.1rango)
    x_grid = range(domain...,length = nGrid) |> collect
    
    yHat = eval(kr,x_grid)
    yBar = repeat([mean(kr.y)],nGrid)

    z = quantile(Normal(),1-alpha/2)
    c = z .*sqrt.(HoVar(kr,x_grid))
    

    p = scatter(kr.x,kr.y,
        label = "Data",
        title = "No Effect Test";
        args...
    )
    p = plot!(p,x_grid,yHat,
        label = "Regression"
    )

    p = plot!(p, x_grid, yBar,
        ribbon = c,
        label = "C. Band"
    )

    return p

end



