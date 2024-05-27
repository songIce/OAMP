
clear
clc

alphagrid=0.1:0.1:3;
prior = "2Points"; % 'Rad': Rademacher; 'Gauss'; 'sparse', '2Points'
alpha1 = 1/2; % parameter for 2Points prior
r = 1; % sparsity
%Set fixed point iterations parameters:
conv=1e-10;
damp=0.0;

% pdf of sestic eigenvalue distribution
func_rho = @(a,xi,x) (6*a^4*xi+2*a^2*xi*x.^2+xi*x.^4).*sqrt(4*a^2-x.^2)./(2*pi);
% a=(1/10/xi)^(1/6); % parameter
xi = 27/80;
a = sqrt(2/3);

m_Rad=zeros(length(alphagrid),2);
variance=integral(@(D) func_rho(a,xi,D).*D.^2,-2*a,2*a);
fprintf("a=%e, %e ==1, second moment check \n",a,variance);

for jj = 1:length(alphagrid)

    lam = alphagrid(jj);
    m0=0.01 + rand(1)/10;
    k10=0.01 + rand(1)/10;
    k20=0.01 + rand(1)/10;
    k30=0.01 + rand(1)/10;
    [m,k1,k2,k3] = FPrecursion(lam,xi,m0,k10,k20,k30,conv,damp,prior,r,alpha1);
    
    m_Rad(jj,:) = [m,k1];
    fprintf("SNR=%e, m=%e \n",lam,m_Rad(jj,1));

end

Replica_MSE_bo = 1-m_Rad(:,1);
Replica_Overlap = m_Rad(:,1);

semilogy(alphagrid,Replica_Overlap)
xlabel("SNR")
ylabel("MSE")

Replica_alphagrid = alphagrid;
save replica_BO_sestic_0_3.mat Replica_alphagrid Replica_MSE_bo;






%% function

function [m,k1,k2,k3] = FPrecursion(lam,xi,m,k1,k2,k3,conv,damp,prior,r,alpha1)

    func_deneig = @(xi,lam,D) -(-xi*lam*D.^5 + xi*lam^2*D.^4 + xi*lam^2*D.^2); % J(lam)
    func_rho = @(a,xi,x) (6*a^4*xi+2*a^2*xi*x.^2+xi*x.^4).*sqrt(4*a^2-x.^2)./(2*pi);
    func_H = @(xi,lam,D,tilde_v) tilde_v - xi*lam*D.^5 + xi*lam^2*D.^4 + xi*lam^2*D.^2;
    func_Q = @(lam,D,tilde_m,m,k1,k2,k3) xi*lam^2*(m*D.^4 + k1*D.^3 + k2*D.^2 + k3*D) - tilde_m;

    crit=conv+1;
    it=0;
    %"a" is chosen such that the spectral density has unit variance
    a = (1/10/xi)^(1/6); %boundary of the spectrum
%     a = sqrt(2/3)-1e-3;

    while crit>conv && it<200
                
        FPtilde_v = @(V) 1-m-integral(@(D) func_rho(a,xi,D)./func_H(xi,lam,D,V),-2*a,2*a);
        %if equated to 0 yields the equation for \tilde{V} in the Supplementary Information
        tilde_v_up=0.0000001*randn(1)+max([func_deneig(xi,lam,2*a),func_deneig(xi,lam,-2*a),-fminbnd(@(D) -func_deneig(xi,lam,D),-2*a,2*a)]);
        %tilde_v_up is chosen to be close to the right boundary of the spectrum spanned by deneig.
        tilde_v=bisection(FPtilde_v,tilde_v_up,tilde_v_up+1000000,128); 
        %the solution to the equation above grows exponentially in \lambda, that is why the left boundary of the dichotomic search is at tilde_v_up+100000000.

        tilde_m=(-m+integral(@(D) func_rho(a,xi,D).*xi*lam^2.*(m*D.^4+k1*D.^3+k2*D.^2+k3*D)./func_H(xi,lam,D,tilde_v),-2*a,2*a))/(1-m);
        %not really needed but it allows to compactify a bit
        hat_m=m/(1-m)+tilde_m+ xi*lam^2*integral(@(D) func_rho(a,xi,D).*(D.^4).*func_Q(lam,D,tilde_m,m,k1,k2,k3)./func_H(xi,lam,D,tilde_v),-2*a,2*a);
        if prior=="Rad"
            m_new=integral(@(Z) exp(-Z.^2/2)./sqrt(2*pi).*tanh(Z*sqrt(hat_m)+hat_m),-15,15);    
        elseif prior=="gauss"
            m_new=hat_m/(1+hat_m);
        elseif prior=="sparse"
            aux=@(X) X*integral(@(Z) exp(-Z.^2/2)./sqrt(2*pi)*sqrt(r).*sinh((Z.*sqrt(hat_m)+hat_m.*X)./sqrt(r))*exp(-hat_m/(2*r))./(1-r+r*cosh((Z.*sqrt(hat_m)+hat_m.*X)./sqrt(r))*exp(-hat_m/(2*r))),-15,15);
            m_new=r/2*(aux(-1/sqrt(r))+aux(1/sqrt(r))) + (1-r)*aux(0);
        elseif prior=="2Points"
            aux1 = @(X,Z) exp(hat_m/2/alpha1^2 - (Z.*sqrt(hat_m)+X.*hat_m)./alpha1 );
            aux=@(X) integral(@(Z) X.*normpdf(Z).*alpha1./( (1 - alpha1^2).*aux1(X,Z) + alpha1^2 ), -Inf, Inf);
            m_new = (alpha1^2.*aux(1/alpha1) + (1-alpha1^2).*aux(0) );
        end
        k1_new=integral(@(D) func_rho(a,xi,D).*D.*func_Q(lam,D,tilde_m,m,k1,k2,k3)./func_H(xi,lam,D,tilde_v),-2*a,2*a);
        k2_new=integral(@(D) func_rho(a,xi,D).*D.^2.*func_Q(lam,D,tilde_m,m,k1,k2,k3)./func_H(xi,lam,D,tilde_v),-2*a,2*a);
        k3_new=integral(@(D) func_rho(a,xi,D).*D.^3.*func_Q(lam,D,tilde_m,m,k1,k2,k3)./func_H(xi,lam,D,tilde_v),-2*a,2*a);

        crit=abs(m_new-m)+abs(k1_new-k1);
        m=(1-damp)*m_new+damp*m;
        k1=(1-damp)*k1_new+damp*k1;
        k2=(1-damp)*k2_new+damp*k2;
        k3=(1-damp)*k3_new+damp*k3;
        
        it=it+1;
        if mod(it,50)==0
          fprintf("it = %d, crit=%e,\n m=%e, k=%e \n",it, crit, m, k1 ) 
        end
    end

    fprintf("v:%e , cal:%e \n", tilde_v, 1/(1-m) - hat_m)

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

