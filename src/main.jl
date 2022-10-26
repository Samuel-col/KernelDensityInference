# inferenciaDensidadKernel
module KernelTests

    export KernelDensity, BivariateKernelDensity
    export density, discretize, plot, plot!
    export sameDistributionTest, independencyTest
    export KernelRegression, eval, residuals, sig2, pointVariance
    export noEffectTest, display, noEffectGraphicTest

    include("baseFunctions.jl")

    include("kernelDensity.jl")

    include("hypothesysTests.jl")

    include("kernelRegression.jl")

    include("regressionInference.jl")

    include("graphicMethods.jl")


end