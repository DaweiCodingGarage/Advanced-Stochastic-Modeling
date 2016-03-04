module Wald

using PyPlot, Distributions

theta0 = 8
a,b = 1,1/8
#y = [7,9,15,1,6,12]
#I = Bool[0,1,0,1,1,0]
y = [7,3,10,7,8,8]
I = Bool[0,1,0,1,0,1]
n = length(y)
nmis = n-sum(I)
myc = mean(y)
myo = mean(y[I])

figure(1); clf(); hold(true)
plot([theta0,theta0],ylim(),"k--",label="true mean",linewidth=2)
plot([myc,myc],ylim(),"g-",label="complete sample mean",linewidth=2)
plot([myo,myo],ylim(),"r-",label="observed sample mean",linewidth=2)

phi(y) = 1.0./sqrt(y+1) # prob of returning
f(y,theta) = pdf(Poisson(theta),y)
prior(theta) = pdf(Gamma(a,1/b),theta)
wposterior(theta) = pdf(Gamma(a+sum(y[I]),1/(b+sum(I))),theta)
ys = 0:200
Pnoreturn(theta) = [(sum(f(ys,th).*(1-phi(ys))))::Float64 for th in theta]
rposterior(theta) = (dt=theta[2]-theta[1]; p=wposterior(theta).*(Pnoreturn(theta).^nmis); p./(dt*sum(p)))

ts = linspace(0,20,5000)
plot(ts,prior(ts),"c-",label="prior",linewidth=2)
plot(ts,wposterior(ts),"y-",label="wrong posterior",linewidth=2)
plot(ts,rposterior(ts),"b-",label="right posterior",linewidth=2)
legend(loc="upper right")
ylim(0,.5)



end # module



