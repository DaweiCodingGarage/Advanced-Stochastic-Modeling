# Code used to generate examples in notes accompanying BDA4.
module BDA4

using Distributions
using PyPlot

example = 1  # which example to run

# Consistency and asymptotic normality
# Model:
#     theta ~ Exp(1)
#     x_1,...,x_n|theta ~ Exp(theta)
# Data: x_1,...,x_n ~ Exp(1)
# The posterior concentrates at the true theta and is asymptotically normal.
if example==1
    ns = [1,10,100,1000]
    colors = ["c","b","g","r"]
    figure(1,figsize=(8,4)); clf(); hold(true); subplots_adjust(bottom=0.15)
    nns = length(ns)
    ds = zeros(nns)
    for (i_n,n) in enumerate(ns)
        x = rand(Exponential(1),n)
        thetas = linspace(0,2,5000)
        posterior = Gamma(n+1,1/(1+sum(x)))
        figure(1); plot(thetas,pdf(posterior,thetas),label="n = $n",linewidth=2,color=colors[i_n])
        N = Normal(mean(posterior),std(posterior))
        xs = linspace(0,10,10000)
        ds[i_n] = maximum(abs(cdf(N,xs) - cdf(posterior,xs)))
    end
    figure(1)
    legend()
    title("Posterior density",fontsize=18)
    xlabel(L"\theta",fontsize=18)
    ylabel(L"p(\theta|x)",fontsize=18)
    draw()
    #savefig("ex$example-pdf",dpi=150)

    figure(2,figsize=(8,4)); clf(); hold(true); subplots_adjust(bottom=0.2)
    title("KS distance from best normal approximation",fontsize=18)
    xlabel("n  (sample size)",fontsize=15)
    ylabel("distance",fontsize=15)
    semilogx(ns,ds,"bo-")
    ylim(0,ylim()[2])
    draw()
    #savefig("ex$example-dist",dpi=150)
end


# Under-identified models and non-identified parameters
# (Note: The example in the textbook is too trivial, so this is different.)
# Model:
#     a,b ~ Exp(1)
#     x_1,...,x_n|a,b ~ Exp(a*b)
# Data: x_1,...,x_n ~ Exp(1)
# Use Gibbs sampling to draw a,b|x_1,...,x_n.
# Any values of a and b such that a*b==1 will fit the data. The posterior is not asymptotically normal.
# (This also serves as an example in which Gibbs sampling mixes poorly for large n.)
if example==2
    a=b=1
    nreps = 100000
    ns = [1,10,100,1000]
    for (i_n,n) in enumerate(ns)
        x = rand(Exponential(1),n)
        as = zeros(nreps)
        bs = zeros(nreps)
        sum_x = sum(x)
        for r = 1:nreps
            a = rand(Gamma(n+1, 1/(b*sum_x+1)))
            b = rand(Gamma(n+1, 1/(a*sum_x+1)))
            as[r] = a
            bs[r] = b
        end
        figure(i_n,figsize=(4,4)); clf(); hold(true)
        subplots_adjust(bottom=0.15)
        plot(as,bs,".",markersize=.1)
        title("n = $n")
        xlabel("a")
        ylabel("b")
        xlim(0,7)
        ylim(0,7)
        draw()
        #savefig("ex$example-n=$n.png",dpi=150)
    end
end

# Confidence intervals and credible intervals
# Model:
#     p ~ Beta(a,b)
#     x_1,...,x_n|p ~ Bernoulli(p)
# Data: x_1,...,x_n ~ Bernoulli(0.1)
if example==3
    p0 = 0.1
    a,b = 1,1
    ns = [1,10,100,1000]
    nns = length(ns)
    nreps = 1000
    methods = ["Wald confidence interval","Central credible interval"]
    colors = ["r","b"]
    markers = ["s","o"]
    nmethods = length(methods)
    counts = zeros(nns,nmethods)
    figure(1,figsize=(8,4)); clf(); hold(true); subplots_adjust(bottom=0.2)
    for (i_n,n) in enumerate(ns)
        intervals = zeros(2,nmethods)
        for r = 1:nreps
            x = rand(Bernoulli(p0),n)
            
            # 95% Wald-type confidence interval (based on normal approx)
            p_hat = mean(x)
            intervals[:,1] = p_hat + [-1,1]*1.96*sqrt(p_hat*(1-p_hat)/n)

            # 95% central credible interval
            posterior = Beta(a+sum(x),b+n-sum(x))
            intervals[:,2] = quantile(posterior,[.025,.975])

            for m = 1:nmethods
                counts[i_n,m] += (intervals[1,m] <= p0 <= intervals[2,m])
            end
        end

        for m = 1:nmethods
            v = (nns+1-i_n)+(m-1.5)/6
            l = (i_n==1? methods[m] : "")
            plot([intervals[1,m],intervals[2,m]],[v,v],label=l,marker=markers[m],linewidth=3,color=colors[m])
        end
        ylim(0,nns+1)
        xlim(-0.2,1)
        plot([p0,p0],ylim(),"k--",linewidth=2)
        yticks(nns:-1:1,ns)
        legend(loc="lower right",numpoints=1)
        title("Typical intervals",fontsize=18)
        xlabel(L"p",fontsize=18)
        ylabel("sample size",fontsize=15)
        draw()
        #savefig("ex$example-intervals",dpi=150)
    end

    figure(2,figsize=(8,4)); clf(); hold(true); subplots_adjust(bottom=0.2)
    coverage = counts/nreps
    for m = 1:nmethods
        semilogx(ns,coverage[:,m],label=methods[m],color=colors[m],marker=markers[m],linewidth=2)
    end
    plot([ns[1],ns[end]],[.95,.95],"k--",linewidth=2)
    legend(loc="lower right",numpoints=1)
    title("Frequentist coverage",fontsize=18)
    xlabel("n  (sample size)",fontsize=18)
    draw()
    #savefig("ex$example-coverage",dpi=150)
end



end # module

nothing




