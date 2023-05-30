#! tutorial ii

using Random
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using Optim

Random.seed!(243587)

# load the data
data = CSV.read("data_ii.csv", DataFrame)

const X = Matrix(data[!, [:x1, :x2]])
const y = Vector(data[!, :y])

# draws of normal random numbers
const M = 10
const u = rand(Normal(0, sqrt(1.5)), length(y), M)

# define auxiliary model estimator
function ols(x, y; add_constant=true)
    if add_constant == true
        n = size(x, 1)
        x = hcat(ones(n), x)
    end
    x \ y # this is the same as (x'*x)^{-1} * x'*y, but faster and more accurate
end

# estimate aux model with observed data
const β̂ = ols(X, y)

# define structural model
function G(X, u, θ)
    n = size(X, 1)
    exp.(hcat(ones(n), X) * θ) + u
end

# define criterion for ii
wald(β̃; β̂=β̂) = sum((β̂ - β̃).^2)

# define objective function
function obj(θ; X=X, u=u, β̂=β̂)
    ỹ = [G(X, u[:, m], θ) for m in 1:M] # this simulates the data
    β̃ = mean(ols(X, ỹ[m]) for m in 1:M) # this estimates aux model on simulated data, and then averages parameters
    return wald(β̃; β̂=β̂)
end

# optimization of criterion
θ0 = ols(X, y) # initial vector of parameters
opt = optimize(θ -> obj(θ; X=X, u=u, β̂=β̂), θ0, NelderMead())
opt.minimizer

# the true values are
#=
 0.9279139553540544
 0.2408481106524355
 0.4354059904502885
=#
