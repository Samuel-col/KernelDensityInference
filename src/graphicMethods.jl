# Métodos gráficos para inferenciaDensidadKernel
using Distributions

## Método para graficar KernelDensity
function plot(KD::KernelDensity;args...)::Plots.Plot
    plot(x -> density(x,KD), KD.domain[1],KD.domain[2];args...)
end

function plot!(KD::KernelDensity;args...)::Plots.Plot
    plot!(x -> density(x,KD), KD.domain[1],KD.domain[2];args...)
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

## Método para graficar KernelRegression
function plot(kr::KernelRegression;se = true,alpha = 0.05,nGrid::Int = 160,args...)::Plots.Plot
    a,b = minimum(kr.x), maximum(kr.x)
    rango = b-a
    domain = (a-0.1rango,b+0.1rango)
    x_grid = range(domain...,length = nGrid) |> collect
    
    yHat = eval(kr,x_grid)
    if se
        yHatVar = pointVariance(kr,x_grid)

        z = quantile(Normal(),1-alpha/2)
        c = z*sqrt.(yHatVar)

        p = plot(x_grid,yHat,ribbon = c, fillalpha = 0.2;args...)
    else
        p = plot(x_grid,yHat;args...)
    end

    return p
end

function plot!(kr::KernelRegression;se = true,alpha::Float64 = 0.05,nGrid::Int = 160,args...)::Plots.Plot
    a,b = minimum(kr.x), maximum(kr.x)
    rango = b-a
    domain = (a-0.1rango,b+0.1rango)
    x_grid = range(domain...,length = nGrid) |> collect
    
    yHat = eval(kr,x_grid)
    if se
        yHatVar = pointVariance(kr,x_grid)

        z = quantile(Normal(),1-alpha/2)
        c = z*sqrt.(yHatVar)

        p = plot!(x_grid,yHat,ribbon = c, fillalpha = 0.2;args...)
    else
        p = plot!(x_grid,yHat;args...)
    end

    return p
end



