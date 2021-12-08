
#MCMC includes distributions library
include("MCMC.jl");

using Plots

###### Symmetry Breaking (Square Well) ######

#Physics Setup
g = 6.0; μ = 1.0
ϕ₀ = 5.0;
Wp(g,ϕ,μ) = g*(ϕ^2 + μ^2)
Wpp(g,ϕ) = 2g*ϕ
S(Wp,Wpp) = (1/2)*Wp^2 - log(abs(Wpp))
Sp = θ -> S(Wp(g,θ,μ),Wpp(g,θ)) 

#Optimization Setup
ϵ = 0.1;
Δ(ϵ) = rand(Normal(0,ϵ)) 
Δp = () -> Δ(ϵ)

#Optmize
ϕs = MCMCoptimize(Sp,ϕ₀,Δp,1000)
plot(ϕs)

#Statistics (Observables)
therm = 250
Expϕ = mean(ϕs[therm:end])
(1/2)*Wp(g,Expϕ,μ)^2

begin #setup
    Therm = 10000
    Sweeps = 10000
    gap = 10
    δ = 0.5

    T = 64
    ω = 0.5; m = 1.0

    site = rand(T) .- 0.5
    oldSite = fill(0.0,T); newSite = fill(0.0,T)
    corr = fill(0.0,T)
    xsq = 0.0; xsq_sq = 0.0
    xVal = 0.0; xValsq = 0.0
end

for i in 1:Therm
    for t in 1:T
        tau = rand(1:T);
        oldSite[tau] = site[tau];
        shift = 2.0*δ*(rand()-0.5)
        newSite[tau] = site[tau] + shift
        ntau = tau + 1
        (ntau > T) && (ntau = 1)
        dS = (m/2)*((site[ntau] - newSite[tau])^2 +
             0.25*(ω^2)*(site[ntau] + newSite[tau])^2 -
             (site[ntau]-oldSite[tau])^2 + 
             0.25*(ω^2)*(site[ntau] + oldSite[tau])^2);
        if min(1,exp(-dS)) > rand()
            site[tau] = newSite[tau]
        else
            site[tau] = oldSite[tau]
        end
    end
end

for i in 1:Sweeps
    for t in 1:T
        tau = rand(1:T);
        oldSite[tau] = site[tau];
        shift = 2.0*δ*(rand()-0.5)
        newSite[tau] = site[tau] + shift
        ntau = tau + 1
        (ntau > T) && (ntau = 1)
        dS = (m/2)*((site[ntau] - newSite[tau])^2 +
             0.25*(ω^2)*(site[ntau] + newSite[tau])^2 -
             (site[ntau]-oldSite[tau])^2 + 
             0.25*(ω^2)*(site[ntau] + oldSite[tau])^2);
        if min(1,exp(-dS)) > rand()
            site[tau] = newSite[tau]
        else
            site[tau] = oldSite[tau]
        end

    end
    if i%gap == 0
        for t in 1:T
            corr[t] = corr[t] + site[t]*site[1]/(2.0*m*ω)
        end
    end
end

plot(site)
scatter(corr/(Sweeps/gap))

##### Hamiltonian MC #####
include("MCMC.jl");

using Plots

begin #setup
    sweeps = 10000;
    L = 20;
    ϵ = 0.2;
    θ₀ = [2.0,2.0];

    g = 6.0; μ = 1.0
    ϕ₀ = 5.0;
    Wp(g,ϕ,μ) = g*(ϕ'ϕ + μ'μ)
    Wpp(g,ϕ) = 2g*ϕ
    S2(Wp,Wpp) = (1/2)*Wp^2 - log(abs(Wpp))
    Sp = θ -> S2(Wp(g,θ,μ),Wpp(g,θ))
    dS2(θ) = 36.0*θ*(θ^2 + 1.0) - 1/(2*6.0*θ) 

    S(ϕ) = 0.5*ϕ'ϕ
    dS(ϕ) = ϕ #force
    #H(ϕ,p) = S(ϕ) + 0.5*p^2
end


θs = MCMChamiltonian(θ₀,S,dS,sweeps,L,ϵ)

data = hcat(θs...)
data2 = (data[1,:],data[2,:])

histogram2d(data2)

θsa = MCMChamiltonian(0.0,S,sweeps,L,ϵ)

