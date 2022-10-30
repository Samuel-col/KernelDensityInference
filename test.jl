# test File para inferenciaDensidadKernel

cd("/home/samuel/Documentos/U/No Paramétrica/inferenciaDensidadKernel/")
include("src/main.jl")
using .KernelTests

# -------------------------- Densidad Kernel ------------------
## Univariada ----


x = randn(1000) .* 5 .+ 3 # Muestra aleatoria N = 1000, μ = 3, σ = 5

kd = KernelDensity(x)

KernelTests.density(0,kd)

KernelTests.density([-10,0,10],kd)

using Plots
gr(minorgrid=true)
p = plot(kd,
    title="Kernel density",
    legend = false,
    xlabel = "x"
)

my_x,my_y = discretize(kd)
plot(my_x,my_y)

## Bivariada ----

x = rand(100)
y = -(x .- 1).*(x .+ 1) .+ 0.1 .*randn(100)

bkd = BivariateKernelDensity(x,y)

plot(bkd,
    title = "Bivariate Kernel Density",
    xlabel = "x",
    ylabel = "y"
)

my_pts,my_dty = discretize(bkd)

typeof(my_pts)
typeof(my_dty)


# -------------------------- Pruebas de Hipótesis ------------------
## Igualdad de distribuciones ----

x = randn(100) .* 1.5 .+ 3 
y = randn(80) .+ 3 

my_test = sameDistributionTest(x,y)

typeof(my_test)
plot(my_test)

## Independencia de poblaciones ----
x = randn(30)
y = exp.(2 .+ rand(30))

my_test = independencyTest(x,y)

typeof(my_test)
plot(my_test)

# -------------------------- Regresión Kernel -----------------------

x = rand(100)

y = 1 ./(0.2 .+ x) .+ 0.4 .*randn(100)

using Plots
gr(minorgrid = true)
scatter(x,y,
    label = "Data",
    title = "Random Data",
    xlabel = "x",
    ylabel = "y",
    legend = :topright
)

kr = KernelRegression(x,y)

KernelTests.eval(kr,0.3)

KernelTests.residuals(kr)

plot!(kr,label = "Regression")

## ---------------------- Pruebas de Hipótesis ----------------
## -- Test de no efecto

my_test = noEffectTest(x,y)

plot(my_test)

my_test = noEffectTest(x,rand(100))

plot(my_test)

noEffectGraphicTest(x,y)

## -- Test de efecto lineal

my_test = linearEffectTest(x,y)

plot(my_test)


linearEffectGraphicTest(x,y)