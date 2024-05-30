% state evolution of OAMP, Some notations differ from those in the paper.

clear all;
clc;
close all;

Iteration = 200;

PCA_init = "F";
% initial omega in [0,1]
omega_SE = 0.0;

% Input snr, Y=theta/N*x*xT + W
Thetagrid = 0.1 : 0.01 : 3; % for SE

prior = "2Points"; % "Rad", "sparse", "3Points"
sparsity = 0.02; % for sparse
alpha1 = 1/2;    % for 2Points and 3Points
alpha2 = -1/10;

% Eigenvalue distribution 
Eig_dist = "quartic"; % "quartic", "sestic"
gamma = 0; % quartic parameter, mu=1 is wigner distribution

%%  eigenvalue distribution 
if Eig_dist == "quartic"
    kappa = (8-9*gamma+sqrt(64-144*gamma+108*gamma.^2-27*gamma.^3))/27;
    xi = 0;
    %regularization to avoid dividing by 0
    if gamma == 1
        a2 = 1;
    else
        a2 = (sqrt(gamma.^2+12*kappa)-gamma)./(6*kappa);
    end
    rhofun = @(y) (gamma+kappa*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi); 
elseif Eig_dist == "sestic"
    % In Jean's paper, they consider pure sestic distirbution, u, gamma, xi
    % are fixed
    gamma = 0;
    kappa = 0;
    xi = 27/80;
    a2 = 2/3;
    rhofun = @(y) (gamma+2*a2*kappa+6*a2^2*xi+(kappa+2*a2*xi)*y.^2+xi*y.^4).*sqrt(4*a2-y.^2)/(2*pi);
end

%% MSE
% || xxT - rrT ||_F
MSE = @(x,r,n) norm(r-x)^2/n;
OAMP_MSE = zeros(Iteration,length(Thetagrid));

OAMP_MSE_cal = zeros(Iteration,length(Thetagrid));
OAMP_overlap_cal = zeros(Iteration,length(Thetagrid));

% Define MMSE function
if prior=="Rad"
    MMSE_SE_func = @(snr_LE) 1 - integral(@(x) normpdf(x).*tanh(snr_LE-sqrt(snr_LE)*x), -inf,inf ); 
elseif prior=="gauss"
    MMSE_SE_func = @(snr_LE) 1 - snr_LE/(1+snr_LE);
elseif prior=="sparse"
    aux=@(X,snr_LE) X*integral(@(Z) normpdf(Z).*sqrt(sparsity).*sinh((Z.*sqrt(snr_LE)+X.*snr_LE)./sqrt(sparsity))*exp(-1/(2*sparsity)*snr_LE)./(1-sparsity+sparsity*cosh((Z.*sqrt(snr_LE)+X.*snr_LE)./sqrt(sparsity))*exp(-snr_LE/(2*sparsity))),-16,16);
    MMSE_SE_func = @(snr_LE) 1 - ( sparsity/2*(aux(-1/sqrt(sparsity),snr_LE)+aux(1/sqrt(sparsity),snr_LE)) + (1-sparsity)*aux(0,snr_LE) );
elseif prior =="3Points"
    aux1 = @(X,Z,snr_LE) exp(-snr_LE/2/alpha1^2 + (Z.*sqrt(snr_LE)+X.*snr_LE)./alpha1 );
    aux2 = @(X,Z,snr_LE) exp(-snr_LE/2/alpha2^2 + (Z.*sqrt(snr_LE)+X.*snr_LE)./alpha2 );
    aux=@(X,snr_LE) integral(@(Z) X.*normpdf(Z).*( alpha1/2.*aux1(X,Z,snr_LE) + alpha2/2.*aux2(X,Z,snr_LE) )./( 1 - alpha1^2/2 - alpha2^2/2 + alpha1^2/2.*aux1(X,Z,snr_LE) + alpha2^2/2.*aux2(X,Z,snr_LE) ), -20, 20);
    MMSE_SE_func = @(snr_LE) 1 - (alpha1^2/2.*aux(1/alpha1,snr_LE) + alpha2^2/2.*aux(1/alpha2,snr_LE));
elseif prior=="2Points"
    aux1 = @(X,Z,snr) exp(-snr/2/alpha1^2 + (Z.*sqrt(snr)+X.*snr)./alpha1 );
    aux=@(X,snr) integral(@(Z) X.*normpdf(Z).*( alpha1.*aux1(X,Z,snr) )./( 1 - alpha1^2 + alpha1^2.*aux1(X,Z,snr) ), -20, 20);
    MMSE_SE_func = @(snr) 1 - (alpha1^2.*aux(1/alpha1,snr) + (1-alpha1^2 ).*aux(0,snr) );
end


%% Testing SE
for itest = 1:length(Thetagrid)
    
    Theta = Thetagrid(itest);
    if PCA_init =="T"
        [~,omega_SE] = Max_eig(Theta,gamma,xi);
    elseif MMSE_SE_func(0) < 1 - 1e-6
        omega_SE = 0;
    else
        snr_LE_SE = omega/(1-omega);
    end
    fprintf("Theta = %.3f, omega = %.3f \n",Theta,omega_SE)
    
    %% se function
    if Eig_dist=="quartic"
        J_Y = @(Y) gamma*Theta*Y-kappa*Theta^2*Y.^2+kappa*Theta*Y.^3;
        J_const = Theta^2*a2*(gamma+2*a2*kappa)^2 + 1;
    elseif Eig_dist=="sestic"
        J_Y = @(Y) xi*Theta*Y.^5-xi*Theta^2*Y.^4-xi*Theta^2*Y.^2;
        J_const = 27/50*Theta^2 + 1;
    end
    P_func = @(rho,Y) 1./(rho + J_const - J_Y(Y)); 
    % we define snr here, it equals to omega/(1-omega) in paper, the
    % details refer to OAMP_demo.mlx file
    Phi_snr = @(rho) 1./integral(@(lam) P_func(rho,lam).*rhofun(lam),-2*sqrt(a2),2*sqrt(a2)) - (rho+1);

    %% State evolution
    
    snr_LE_it = zeros(Iteration,1);
    v_opt = zeros(Iteration,1);
    snr_LE = omega_SE/(1-omega_SE); % snr of the LE 
    
    for it = 1:Iteration

        
        MMSE_SE = MMSE_SE_func(snr_LE);
        
        dmmse = 1/( 1/MMSE_SE - snr_LE);
        rho = 1/(dmmse) - 1;
        
        overlap_SE = 1-MMSE_SE;

        snr_LE = Phi_snr(rho);

        v_opt(it) = rho;
        snr_LE_it(it) = snr_LE;
    
        OAMP_MSE_cal(it,itest) = MMSE_SE;
        OAMP_overlap_cal(it,itest) = overlap_SE;
        if it > 3 && (OAMP_MSE_cal(it-1,itest)-MMSE_SE) <1e-10
            OAMP_MSE_cal(it+1:end,itest) = OAMP_MSE_cal(it,itest);
            OAMP_overlap_cal(it+1:end,itest) = overlap_SE;
            break;
        end

    end
    
    fprintf("MSE_se:%f \n",OAMP_MSE_cal(it,itest))
    
end


OAMP_SE_Thetagrid = Thetagrid;
OAMP_SE_MSE = OAMP_MSE_cal;
OAMP_SE_overlap = real(OAMP_overlap_cal);

% if Eig_dist == "quartic"
%     filename = strcat("OAMP_BO_quartic_u",num2str(mu),"_theta0_3.mat");
% elseif Eig_dist == "sestic"
%     filename = "OAMP_BO_sestic_theta0_3.mat";
% end
% save(filename,"OAMP_SE_Thetagrid","OAMP_SE_MSE","OAMP_SE_overlap")
% save OAMP_BO_sestic_theta0_3.mat OAMP_SE_Thetagrid OAMP_SE_MSE OAMP_SE_overlap



sjt = @(z) 1/2*(gamma*z+ kappa*z.^3 - ( gamma + 2*a2*kappa + kappa*z.^2 ).*sqrt(z.^2-4*a2));
PCA_PT_lam = 1/sjt(2*sqrt(a2));

plot(OAMP_SE_Thetagrid,OAMP_SE_overlap(end,:));hold on
xline(real(PCA_PT_lam))


%% bisection method for PCA 

function [max_eig_cal,overlap_cal] = Max_eig(Lambda,u,xi)
    
    if xi == 0
        gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;
        if u == 1
            a2 = 1;
        else
            a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
        end
        
        sjt = @(z) 1/2*(u*z+ gamma*z.^3 - ( u + 2*a2*gamma + gamma*z.^2 ).*sqrt(z.^2-4*a2));
        sjt_diff = @(z) 1/2*( u + 3*gamma*z.^2 - 2*gamma*z.*sqrt(z.^2-4*a2) - ...
            z.*( u + 2*a2*gamma + gamma*z.^2 )./sqrt(z.^2-4*a2) );
    else
        a2 = 2/3;
        sjt = @(z) 1/2*(xi*z.^5 - ( 6*a2^2*xi + 2*a2*xi.*z.^2 + xi*z.^4 ).*sqrt(z.^2-4*a2));
        sjt_diff = @(z) 1/2*( 5*xi*z.^4 - ( 4*a2*xi.*z + 4*xi.*z.^3 ).*sqrt(z.^2 - 4*a2) - ...
            ( 6*a2^2*xi + 2*a2*xi.*z.^2 + xi*z.^4 ).*z./sqrt(z.^2-4*a2) );
    end
    if Lambda < real(1/sjt(2*sqrt(a2)))
        max_eig_cal = 2*sqrt(a2);
        overlap_cal = 0;
    else

        x_1 = 0;
        x_2 = 10*Lambda;
        it = 1;
        while it < 1000
            
            max_eig = (x_2 + x_1)/2;
            Lambda_tmp = 1/sjt(max_eig);
            if Lambda_tmp > Lambda
                x_2 = max_eig;
            else
                x_1 = max_eig;
            end
            it = it + 1;
            if abs(x_2-x_1) < 1e-6
                break
            end
        end
        max_eig_cal = (x_1+x_2)/2; 
        overlap_cal = -1/Lambda^2/sjt_diff(max_eig_cal);
    end

end