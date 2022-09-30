# Borrador ejemplo inferenciaDensidadKernel

cd("/home/samuel/Documentos/U/No Paramétrica/inferenciaDensidadKernel/")
include("src/main.jl")
using .KernelTests


# Densidad kernel

x = randn(1000) .* 5 .+ 3 # Muestra aleatoria N = 1000, μ = 3, σ = 5

kd = KernelDensity(x)

KernelTests.density(0,kd)

KernelTests.density([-10,0,10],kd)

## Graficar

using Plots
gr(minorgrid=true)
p = plot(kd,
    title="Kernel density",
    legend = false,
    xlabel = "x"
)

my_x,my_y = discretize(kd)
plot(my_x,my_y)


# Prueba de igualdad de distribuciones

x = randn(100) .* 1.5 .+ 3 
y = randn(80) .+ 3 

plot(discretize(KernelDensity(x))...,legend = :topright,label = "x")
plot!(discretize(KernelDensity(y))...,label = "y")

my_test = sameDistributionTest(x,y)

plot(my_test)



x = randn(20) .+ 2.7 
y = randn(20) .+ 3 

plot(discretize(KernelDensity(x))...,legend = :topright,label = "x")
plot!(discretize(KernelDensity(y))...,label = "y")

my_test = sameDistributionTest(x,y)

plot(my_test)



x = randn(500) .+ 2.7 
y = randn(500) .+ 3 

plot(discretize(KernelDensity(x))...,legend = :topright,label = "x")
plot!(discretize(KernelDensity(y))...,label = "y")

my_test = sameDistributionTest(x,y)

plot(my_test)


# Prueba de independencia de variables

x = randn(30)
y = exp.(2 .+ rand(30))

my_test = independencyTest(x,y)

plot(my_test)



x = randn(30)
y = exp.(2 .+ rand(30)) .+  4x

my_test = independencyTest(x,y)

plot(my_test)




# Kernel bivariado

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

independencyTest(x,y)
independencyTest(x,y).plot


plot(KernelDensity(x))
plot!(KernelDensity(y))