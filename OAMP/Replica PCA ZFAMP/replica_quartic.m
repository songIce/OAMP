% This code was used to generate replica result
% 
clear
clc

alphagrid=0.6:0.001:0.9;
prior = "sparse"; % 'Rad': Rademacher; 'Gauss'; 'sparse': 
r = 0.05; % sparsity

%Set fixed point iterations parameters:
conv=1e-10;
damp=0.0;

% pdf of quartic eigenvalue distribution
func_rho = @(gamma,mu,a,x) (mu+2*gamma*a^2+gamma*(x.^2)).*sqrt(4.*a.^2-x.^2)./(2*pi);

m_Rad=zeros(length(alphagrid),2);
mu = 0;
gamma=1/27*(8-9*mu+sqrt(64-144*mu+108*mu^2-27*mu^3));
c=sqrt((sqrt(mu^2+12*gamma)-mu)/(6*gamma));
variance=integral(@(D) func_rho(gamma,mu,c,D).*D.^2,-2*c,2*c);
fprintf("a=%e, %e ==1, second moment check \n",gamma,variance);
for jj = 1:length(alphagrid)

    lam = alphagrid(jj);
    m0=0.01 + rand(1)/10;
    k0=0.01 + rand(1)/10;
    [m,k] = FPrecursion(lam,mu,m0,k0,conv,damp,prior,r);
    
    m_Rad(jj,:) = [m,k];
    fprintf("SNR=%e, m=%e \n",lam,m_Rad(jj,1));

end

Replica_MSE_bo = 1-m_Rad(:,1);

semilogy(alphagrid,Replica_MSE_bo)
xlabel("SNR")
ylabel("MSE")


Replica_alphagrid = alphagrid;
Replica_u = mu;
save replica_BO_u0_0_3.mat Replica_alphagrid Replica_MSE_bo Replica_u;




%% function

function [m,k] = FPrecursion(lam,mu,m,k,conv,damp,prior,r)

    func_deneig = @(gamma,lam,D,mu) -(gamma*lam*(lam*D^2-D^3)-mu*lam*D);
    func_rho = @(gamma,mu,a,x) (mu+2*gamma*a^2+gamma*(x.^2)).*sqrt(4.*a.^2-x.^2)./(2*pi);
    func_H = @(gamma,lam,D,tilde_v,mu) gamma*lam*(lam*D.^2-D.^3)-mu*lam*D+tilde_v;
    func_Q = @(gamma,lam,D,tilde_m,m,k) gamma*m*(lam*D).^2+gamma*lam.^2*k*D+tilde_m;

    crit=conv+1;
    it=0;
    gamma=1/27*(8-9*mu+sqrt(64-144*mu+108*mu^2-27*mu^3));
    %"a" is chosen such that the spectral density has unit variance
        
    while crit>conv && it<200
            
        
        c_hat=sqrt((sqrt(mu^2+12*gamma)-mu)/(6*gamma)); 
        %boundary of the spectrum
    
        FPtilde_v = @(V) 1-m-integral(@(D) func_rho(gamma,mu,c_hat,D)./func_H(gamma,lam,D,V,mu),-2*c_hat,2*c_hat);
        %if equated to 0 yields the equation for \tilde{V} in the Supplementary Information
        tilde_v_up=0.0001*randn(1)+max([func_deneig(gamma,lam,2*c_hat,mu),func_deneig(gamma,lam,-2*c_hat,mu),-fminbnd(@(D) -func_deneig(gamma,lam,D,mu),-2*c_hat,2*c_hat)]);
        %tilde_v_up is chosen to be close to the right boundary of the spectrum spanned by deneig.
        tilde_v=bisection(FPtilde_v,tilde_v_up,tilde_v_up+10000,64); 
        %the solution to the equation above grows exponentially in \lambda, that is why the left boundary of the dichotomic search is at tilde_v_up+100000000.
        
        tilde_m=(m-integral(@(D) func_rho(gamma,mu,c_hat,D).*(gamma*lam.^2*(m*D.^2+k*D))./func_H(gamma,lam,D,tilde_v,mu),-2*c_hat,2*c_hat))/(1-m);
        %not really needed but it allows to compactify a bit
        hat_m=m/(1-m)-tilde_m+mu*lam^2*m+gamma*(lam^2)*integral(@(D) func_rho(gamma,mu,c_hat,D).*(D.^2).*func_Q(gamma,lam,D,tilde_m,m,k)./func_H(gamma,lam,D,tilde_v,mu),-2*c_hat,2*c_hat);
        
        if prior=="Rad"
            m_new=integral(@(Z) exp(-Z.^2/2)./sqrt(2*pi).*tanh(Z*sqrt(hat_m)+hat_m),-15,15);    
        elseif prior=="gauss"
            m_new=hat_m/(1+hat_m);
        elseif prior=="sparse"
            aux=@(X) X*integral(@(Z) exp(-Z.^2/2)./sqrt(2*pi)*sqrt(r).*sinh((Z.*sqrt(hat_m)+hat_m.*X)./sqrt(r))*exp(-hat_m/(2*r))./(1-r+r*cosh((Z.*sqrt(hat_m)+hat_m.*X)./sqrt(r))*exp(-hat_m/(2*r))),-15,15);
            m_new=r/2*(aux(-1/sqrt(r))+aux(1/sqrt(r))) + (1-r)*aux(0);
        end
        k_new=integral(@(D) func_rho(gamma,mu,c_hat,D).*D.*func_Q(gamma,lam,D,tilde_m,m,k)./func_H(gamma,lam,D,tilde_v,mu),-2*c_hat,2*c_hat);
        
        crit=abs(m_new-m)+abs(k_new-k);
        m=(1-damp)*m_new+damp*m;
        k=(1-damp)*k_new+damp*k;
        
        it=it+1;
        if mod(it,50)==0
          fprintf("it = %d, crit=%e,\n m=%e, k=%e \n",it, crit, m, k ) 
        end
    end

end


function root = bisection(func, x_1, x_2, precision)
    it = 1;
    while it < precision
        
        m = (x_2 + x_1)/2;
        fm = func(m);
        if fm>0
            x_2 = m;
        else
            x_1 = m;
        end
        it = it + 1;
    end
    root = (x_1+x_2)/2;

end

