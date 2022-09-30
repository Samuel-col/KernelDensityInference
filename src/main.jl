# inferenciaDensidadKernel
module KernelTests

    export KernelDensity, BivariateKernelDensity
    export density, discretize, plot, plot!
    export sameDistributionTest, independencyTest

    include("baseFunctions.jl")

    include("kernelDensity.jl")

    include("hypothesysTests.jl")

    include("graphicMethods.jl")


end