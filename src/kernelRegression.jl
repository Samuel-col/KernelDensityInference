# regresión kernel para inferencia densidad kernel

struct KernelRegression <: Any
    S::Matrix{T} where T<:Real
    df::Float64
    x::Vector{T} where T<:Real
    y::Vector{T} where T<:Real
    h::Float64
    n::Int
  
    function KernelRegression(x::Vector{T},y::Vector{T};
                                h::Union{T,Missing} = missing) where T<:Real
        if length(x) != length(y)
            stop("x and y must have the same length.")
        end

        n = length(x)

        if ismissing(h)
            println("Calculating optimum bandwidth using leave-one-out cross-validation...")
            rango = -(minimum(x) - maximum(x))
            #hs = range(1/500*rango,1/10*rango,length = 99)
            hs = range(rango/100,rango/10,length = 23)
            errors = zeros(23)

            @threads for i in 1:23
                results = crossValidation_kr(KernelRegression,x,y,h = hs[i])
                expected = [eval(results[k],x[k]) for k in 1:n]
                results = nothing
                total_error = sum((y .- expected).^2)
                errors[i] = total_error
            end

            h = hs[argmin(errors)]

            println("Optimum bandwidth: $h")
        end
               
        #h = optimum_bandwidth(x)
    
        S = [K(x[j] - x[i],h = h) for j = 1:n, i = 1:n]
        sTotalRow = sum(S,dims = 2)
        S ./= sTotalRow
    
        df = tr((I-S)'*(I-S))
    
        new(S,df,x,y,h,n)
    end
end
    
function eval(kr::KernelRegression,x::T)::Float64 where T<:Real
    w = [K(x - kr.x[j],h = kr.h) for j in 1:(kr.n)]
    return kr.y'w ./sum(w)
end

function eval(kr::KernelRegression,x::Vector{T})::Vector{Float64} where T<:Real
    return [eval(kr,xi) for xi in x]
end

function residuals(kr::KernelRegression)::Vector{Float64}
    yHat = kr.S * kr.y
    return kr.y .- yHat
end

function sig2(kr::KernelRegression)::Float64
    res = residuals(kr)
    return sum(res.^2)/(kr.n - kr.df)
end

function ESS(kr::KernelRegression)::Float64
    res = residuals(kr)
    return sum(res.^2)
end

function sig2(kr::KernelRegression)::Float64
    return ESS(kr)/(kr.n - kr.df)
end

function RSS(kr::KernelRegression)::Float64
    return var(kr.y)*(kr.n-1) - ESS(kr)
end

function Fstat(kr::KernelRegression)::Float64
    cmReg = RSS(kr)/(kr.df - 1)
    cmRes = sig2(kr)
    return cmReg/cmRes
end

function pointVariance(kr::KernelRegression,x::T)::Float64 where T<: Real
    w = [K(x - kr.x[j],h = kr.h) for j in 1:(kr.n)]
    w = w./sum(w)
    return sig2(kr)*sum(w.^2)
end

function pointVariance(kr::KernelRegression,x::Vector{T})::Vector{Float64} where T<: Real
    return [pointVariance(kr,xi) for xi in x]
end

function display(kr::KernelRegression)
    report = """
        Kernel Regresion
          ⋅ Sample size = $(kr.n)
          ⋅ Bandwidth = $(kr.h)
          ⋅ Regression degrees of fredom = $(kr.n - kr.df - 1)
          ⋅ σ² = $(sig2(kr))
        """
    println(report)
end

            