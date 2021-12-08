######## Harmonic Oscillator ########
include("MCMC.jl");
using Plots
using LinearAlgebra
using HypothesisTests
gr()

function energyAvg(data,i,f,a)
    N = f-i+1
    c = data[i:1:f]
    m = mean.(c[a*(i-1)+1:1:a*(i-1)+a] for i in 1:Int(floor(N/a)))
    return var.(m) + mean.(m).^2
end

function waveStats(r,T,d,S)
    rsweeps = r[T:d:S]
    return [hcat(rsweeps...)...]
end

begin #define functions
    function S(r,M,AvgM,dt,m,λ)
        dr = M*r; ar = AvgM * r;
        dt*(KEsum(dr,dt,m) + Vsum(ar))
    end
    KEsum(dr,dt,m) = (m/2)*((dr'dr)/(dt^2));
    function Vharmonic(ar,m,ωsqr,λ)
        ar2 = ar'ar; 
        ((m*ωsqr)/2)*ar2 + λ*ar2^2
    end
    aₖ(k,a,A,α) = a/(k+1+A)^α
    cₖ(k,c,γ) = c/(k+1)^γ
end

begin #setup
    Therm = 1000
    Sweeps = 10000 + Therm
    Nt = 48 #path length
    
    d = ones(Nt)
    u = ones(Nt-1)
    M = Matrix(Bidiagonal(-1*d,u,:U)); M[Nt,1] = 1;
    AvgM = Matrix(Bidiagonal(d,u,:U)); AvgM[Nt,1] = 1; AvgM ./= 2
    dt = 1
    m = 1; ω = 1; λ = 0;
    corrl(r) = [r[t]*r[1]/(2.0)*m*ω for t in 1:Nt]
    Vsum = (dr) -> Vharmonic(dr,m,ω^2,λ)
    a = 2; A = 200; α = 0.9;
    c = 1; γ = 1;
    ak = aₖ.(1:Sweeps,a,A,α)
    ck = cₖ.(1:Sweeps,c,γ)

    A = (r) -> S(r,M,AvgM,dt,m,λ)
    N = MultivariateNormal(zeros(Nt),0.01)
    ΔNorm = () -> sqrt(Nt)*rand(N)
    ΔUnif() = 0.5*(rand(Nt) .- 0.5)
    Δ() = [rand((-1,1)) for i in 1:Nt]
end


begin
    θ₀ = 5*rand(Nt)
    rs1 = MCMC(A,θ₀,ΔUnif,Sweeps)
    rs2 = SGHMC(θ₀,A,Δ,Sweeps,10,0.1,0.02,2)
    #rTerm = last(rs); plot(rTerm)
    ws1 = waveStats(rs1,Therm,1,Sweeps); ws2 = waveStats(rs2,Therm,5,Sweeps);

    println("MCMC Mean: ", mean(ws1)); 
    println("MCMC E₀: ", var(ws1)+mean(ws1)^2);
    println("SGHMC Mean: ", mean(ws2)); 
    println("SGHMC E₀: ", var(ws2)+mean(ws2)^2);
    n = Normal(0,sqrt(0.5))
    x = -4:0.1:4
    plot(x,pdf.(n,x), label = "Schrodinger Solution", 
                      lw = 2, title = "|Ψ₀|² Estimates",
                      xlabel = "x", ylabel = "|Ψ₀|²")
    f1 = fit(Normal,ws1); f2 = fit(Normal,ws2)
    scatter!(x,pdf.(f1,x), label = "MCMC"); 
    scatter!(x,pdf.(f2,x), markershape = :diamond, label = "SGHMC")
end

plot(rs2[end], ylabel = "x", xlabel = "τ", 
               title = "Sample Paths", label = "Sample 1")
plot!(rs2[end-40], label = "Sample 2")
plot!(rs2[end-80], label = "Sample 3")
plot!(rs2[end-120], label = "Sample 4")

var1 = var.(rs1[Therm:1:Sweeps])
mean(var1)
tCI(var1)
var2 = var.(rs2[Therm:1:Sweeps])
mean(var1)
tCI(var2)

histogram(ws1, normalized = true, bins = 60,
title = "|Ψ₀|² MCMC results", label = "MCMC",
xlabel = "x", ylabel = "|Ψ₀|²")
plot!(x,pdf.(n,x), lw = 3, label = "Schrodinger Solution")

histogram(ws2, normalized = true, bins = 60,
title = "|Ψ₀|² SGHMC results", label = "SGHMC",
xlabel = "x", ylabel = "|Ψ₀|²")
plot!(x,pdf.(n,x), lw = 3, label = "Schrodinger Solution")

savefig("samples.png")
begin
    v0 = var(θ₀)
    a = 10
    N = 400
    E1₀ = energyAvg(rs1,1,N*10,10); E2₀ = energyAvg(rs2,1,N,5)
    
    plot([0;N],[0.5,0.5], title = "Energy Estimates", label = "E₀",
                          xlabel = "Number of Function Evaluations",
                          ylabel = "E")
    plot!(1:1:N+1,[v0,E1₀...], label = "MCMC"); 
    plot!(1:5:N+1,[v0,E2₀...], label = "SGHMC");
end

plot(last(rs2))
E1 = energyAvg(rs1,Therm,Sweeps,5); E2 = energyAvg(rs2,Therm,Sweeps,5)
tCI(E1)
tCI(E2)

mean(E1)
mean(E2)
var(E1)
var(E2)
#corr = mean(corrl.(rsweeps))
#scatter(corr)

wf(x,m,ω) = (((m*ω)/(pi))^(1/2))*exp(-m*ω*x^2)

n = Normal(0,sqrt(0.5))
x = -4:0.1:4
plot(x,wf.(x,1,1))
#plot(x,pdf.(n,x))
x = -4:0.1:4
plot(x,wf.(x,1,1))