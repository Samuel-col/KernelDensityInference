# Métodos gráficos para inferenciaDensidadKernel

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