# test File

cd("/home/samuel/Documentos/U/No Paramétrica/")
include("inferenciaDensidadKernel.jl")
using .KernelTests

# Densidad kernel

x = randn(1000) .* 5 .+ 3 # Muestra aleatoria N = 1000, μ = 3, σ = 5

kd = KernelDensity(x)

density(0,kd)

density([-10,0,10],kd)

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

my_test.plot



x = randn(20) .+ 2.7 
y = randn(20) .+ 3 

plot(discretize(KernelDensity(x))...,legend = :topright,label = "x")
plot!(discretize(KernelDensity(y))...,label = "y")

my_test = sameDistributionTest(x,y)

my_test.plot
