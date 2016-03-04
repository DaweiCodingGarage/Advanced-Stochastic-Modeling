module HW

using PyPlot, Distributions, StatsBase

srand(0)

K = 5
alpha = ones(K)
a,b = 1,1
nreps = 10000
n = 5138
s = 20

t0 = [.2,.1,.1,.4,.2]
p0 = [.9,.2,.2,.1,.4]

y0 = rand(Categorical(t0),n)
I = Bool[rand()<p0[yi] for yi in y0]
S = (randperm(n).<=s)
nmis = n-sum(I)
M = find(!(I|S))
nM = length(M)


ysave = copy(y0)
ysave[M] = 0
writedlm("widgets_.txt",[ysave I S])

ts = zeros(nreps,K)
ps = zeros(nreps,K)
t = ones(K)/K
p = ones(K)/2
y = copy(y0)
for rep = 1:nreps
    N = counts(y,1:K)
    R = counts(y.*I,1:K)
    t = rand(Dirichlet(alpha+N))
    for k = 1:K; p[k] = rand(Beta(a+R[k],b+N[k]-R[k])); end
    pM = t.*(1-p) / sum(t.*(1-p))
    y[M] = rand(Categorical(pM),nM)
    ts[rep,:] = t
    ps[rep,:] = p
end

naive = (counts(y[I],1:K)/sum(I))
survey = (counts(y[S],1:K)/sum(S))
tS = rand(Dirichlet(alpha+counts(y[S],1:K)),1000)'

figure(1); clf(); hold(true)
plot(1:K,t0,"b-",linewidth=2)
tm = vec(mean(ts,1))
tsig = vec(std(ts,1))
plot(1:K,tm,"k-",linewidth=2)
plot(1:K,tm + tsig,"k--")
plot(1:K,tm - tsig,"k--")
plot(1:K,naive,"g-",linewidth=2)
tSm = vec(mean(tS,1))
tSsig = vec(std(tS,1))
plot(1:K,tSm,"r-",linewidth=2)
plot(1:K,tSm+tSsig,"r--")
plot(1:K,tSm-tSsig,"r--")


figure(2); clf(); hold(true)
plot(1:K,p0,"b-",linewidth=2)
pm = vec(mean(ps,1))
psig = vec(std(ps,1))
plot(1:K,pm,"k-",linewidth=2)
plot(1:K,pm + psig,"r--")
plot(1:K,pm - psig,"r--")



end # module



