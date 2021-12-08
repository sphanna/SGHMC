using Distributions

function tCI(x,conf_level=0.95)
    N = length(x)
    alpha = (1 - conf_level)
    tstar = quantile(TDist(N-1), 1 - alpha/2)
    r = tstar * std(x)/sqrt(N)
    s = mean(x)
    return [s - r, s + r]
end

function compositeMidpointInt(f,a,b,N)
    h = (b-a)/N
    return h*sum(f(a+(i-0.5)*h) for i in 1:N)
end

function compositeSimpsonInt(f,a,b,N)
    if isodd(N) N = N - 1 end
    h = (b-a)/N
    return (f(a)+f(b))*(h/3.0) +
           4.0*(h/3.0)*sum(f((a+(2i-1)*h)+h) for i in 1:N/2) +
           2.0*(h/3.0)*sum(f(a+(2i*h)+h) for i in 1:N/2-1);
end

function meanErr(f)
    I = sum(f) / N
    I2 = sum(f.^2) / N
    std = sqrt((I2 - I^2)/(N-1))
    return (I,std)
end

function naiveMonteCarloInt(f,a,b,N)
    r = a .+ (b-a)*rand(N)
    fs = f.(r)
    return (b-a) .* meanErr(fs)
end

function importanceMonteCarloInt(f,q,N)
    r = rand(q,N)
    weights = f.(r) ./ pdf.(q,r)
    return meanErr(weights)
end

function standardMonteCarlo(f,X,N)
    XSamples = rand(X,N)
    μ = f.(XSamples)
    return (mean(μ),sqrt(var(μ)/N))
end

#importance sampling of f(X) using proposal distribution q
function importanceMonteCarlo(f,X,q,N)
    qSamples = rand(q,N)
    weights = pdf.(X,qSamples) ./ pdf.(q,qSamples)
    μ = weights .* f.(qSamples)
    return (mean(μ),sqrt(var(μ)/N))
end

MHtest(p,q,y,x) = min(1,(pdf(p,y)/pdf(p,x)*(pdf(q,x)/pdf(q,y)))) > rand()

#Sampling p based on proposal q using Metropolis Hastings algorithm.
#M is number of burn-in iterations (thermalization)
#N is the number of samples after burn-in 
function metropolisHastingsMontecarlo(x₀,p,q,M,N)
    x = x₀;
    #burn-in period: M iterations
    for i in 1:M
        y = rand(q);
        MHtest(p,q,y,x) && (x = y)
    end

    X = fill(x,N+1)
    accepted = 0;
    for i in 1:N
        y = rand(q);
        if MHtest(p,q,y,x) 
            X[i+1] = y
            accepted += 1
        else
            X[i+1] = X[i]
        end
    end
    return (X,accepted/N)
end

function MCMC(S,θ₀,Δ,N)
    θ = fill(θ₀,N+1)
    for i in 1:N
        θnew = θ[i] + Δ()
        dS = S(θnew) - S(θ[i])
        if min(1,exp(-dS)) > rand() 
            θ[i+1] = θnew
        else
            θ[i+1] = θ[i]
        end
    end
    return θ
end

function evolve(ϕ,ϵ,L,S,dS)
    p = rand(MvNormal(length(ϕ),1))
    Hi = S(ϕ) + 0.5*p'p
    ϕ += 0.5*ϵ*p
    for i in 1:L
        p = p - ϵ*dS(ϕ)
        ϕ = ϕ + ϵ*p
    end

    p = p - ϵ*dS(ϕ)
    ϕ = ϕ + ϵ*p

    Hf = S(ϕ) + 0.5*p'p
    return (ϕ,Hf - Hi)
end

function evolveAprx(q,ϵ,L,S)
    p = rand(MvNormal(length(q),1))
    Hi = S(q) + 0.5*p'p
    q = q + 0.5*ϵ*p
    for i in 1:L
        
        p = p - ϵ*q
        q = q + 0.5*ϵ*p
    end
    p = p - ϵ*q
    q = q + 0.5*ϵ*p
    Hf = S(q) + 0.5*p'p
    return (q,Hf - Hi)
end

function SGHMC(θ₀,S,Δ,N,L,c,ϵ,C)
    θ = fill(θ₀,N+1)
    d = length(θ₀)
    
    for k in 1:N
        p = rand(MvNormal(d,1))
        q = θ[k]
        #Hi = S(q) + 0.5*p'p
        for i in 1:L
            q = q + ϵ*p
            p = p - ϵ*gRDSA(S,q,Δ,c) - ϵ*C*p# + sqrt(2ϵ*C)*rand(MvNormal(d,1))
        end
        θ[k+1] = q
        #=Hf = S(q) + 0.5*p'p
        if min(1,exp(-(Hf-Hi))) > rand() 
            θ[k+1] = q
        else
            θ[k+1] = θ[k]
        end=#
    end

    return θ
end

function HMCdS(θ₀,S,dS,N,L,ϵ)
    θ = fill(θ₀,N+1)

    for i in 1:sweeps
        θnew,dH = evolve(θ[i],ϵ,L,S,dS)
        if exp(-dH) > rand()
            θ[i+1] = θnew
        else
            θ[i+1] = θ[i]
        end
    end
    return θ
end

function HMCMH(θ₀,S,N,L,ϵ)
    θ = fill(θ₀,N+1)

    for i in 1:N
        θnew,dH = evolveAprx(θ[i],ϵ,L,S)
        if exp(-dH) > rand()
            θ[i+1] = θnew
        else
            θ[i+1] = θ[i]
        end
    end
    return θ
end

function RDSAMC(y,θ₀,Δ,a,c,N)
    θ = fill(θ₀,N+1)
    Δₖ = [Δ() for k in 1:N]

    for k in 1:N
        dS = gRDSA(y,θ[k],Δₖ[k],c)
        if exp(dS) > rand()
            θ[k+1] = θ[k] - a*dS*Δₖ[k]
        else
            θ[k+1] = θ[k]
        end
    end
    
    return θ
end

function RDSA(y,θ₀,Δ,aₖ::AbstractArray,cₖ::AbstractArray,N)
    θ = fill(θ₀,N+1)
    θ[1] = θ₀
    p = length(θ₀)
    Δₖ = [Δ() for k in 1:N]
    [θ[k+1] = θ[k] - aₖ[k]*gRDSA(y,θ[k],Δₖ[k],cₖ[k]) for k in 1:N]
    return θ
end

function gRDSA(y,θₖ,Δ,cₖ)
    Δr = Δ()
    ckΔ = cₖ*Δr
    (1/(2*cₖ))*(y(θₖ+ckΔ) - y(θₖ-ckΔ)) * Δr
end

#Testing

#=
using Plots

begin
    M = 3000
    N = 1000000

    p = MultivariateNormal([1,0],0.1)
    q = MultivariateNormal(zeros(2),2)

    X,a = metropolisHastingsMontecarlo([0.2,0.2],p,q,M,N)

    a
    mean(X)

    Y = unique(X)

    histogram2d(first.(Y),last.(Y),nbins = 100)

    #histogram(X,normalized = true)
    #x = 0:0.01:10
    #plot!(x,pdf.(p,x))

end
=#

