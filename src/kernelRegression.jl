# regresión kernel para inferencia densidad kernel

struct KernelRegression <: Any
  S::Matrix{T} where T<:Real
  df::Int
  x::Vector{T} where T<:Real
  y::Vector{T} where T<:Real
  
  function KernelDensity(x::Vector{T},y::Vector{T})
    h = optimum_bandwidth(x)
    n = length(x)
    
    S = [K(x[j] - x[i],h = h) for j = 1:n, i = 1:n]
    sTotalRow = sum(S,2)
    S ./= sTotalRow
    
    df = tr((I-S)'(I-S))
    
    new(S,df,x,y)
  end
end
    
