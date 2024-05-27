%% Resolvent approach, optimal v chooses by 1/tau2 + J(lambda1) 
% and comprare with the search one 

clear all;
clc;
close all;

Iteration = 20;
TestNum = 1;
N = 5e5;
D = 5; %order of polynomial

omega = 0.3; % [0,1]; r_0 = sqrt(omega)*x_0 + sqrt(1-omega)*z 


% Input snr, Y=theta/N*x*xT + W
theta = 2;

% Eigenvalue distribution 
Eig_dist = "beta"; % "quartic", "sestic", "beta"
gamma = 0; % quartic parameter, mu=1 is wigner distribution

Prior_x = "Rad"; % "GB", "Rad"
rho = 0.1;

%% MSE
% || xxT - rrT ||_F
MSE = @(x,r,n) 1/n^2 * ( norm(r)^4 + norm(x)^4 - 2*(x'*r)^2 )/2;

for itest = 1:TestNum
    %% Generating eigenvalue    
    tic
    a2 = 1; kappa = 1;
    if Eig_dist == "quartic"
        kappa = (8-9*gamma+sqrt(64-144*gamma+108*gamma.^2-27*gamma.^3))/27;
        %regularization to avoid dividing by 0
        if gamma == 1
            a2 = 1;
        else
            a2 = (sqrt(gamma.^2+12*kappa)-gamma)./(6*kappa);
        end
        rhofun = @(y) (gamma+kappa*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi); 
        d = slicesample(1,N,"pdf",rhofun); % Generating eigenvalues 
    elseif Eig_dist == "sestic"
        % In Jean's paper, they consider pure sestic distirbution, u, gamma, xi
        % are fixed
        gamma = 0;
        kappa = 0;
        xi = 27/80;
        a2 = 2/3;
        rhofun = @(y) (gamma+2*a2*kappa+6*a2^2*xi+(kappa+2*a2*xi)*y.^2+xi*y.^4)*sqrt(4*a2-y.^2)/(2*pi);
        d = slicesample(1,N,"pdf",rhofun); % Generating eigenvalues 
    elseif Eig_dist == "beta"
        beta_a = 1;
        beta_b = 2;
        d = betarnd(beta_a,beta_b,N,1); % Generating eigenvalues 
        d = d - mean(d);
        d = d./sqrt(mean(d.^2));
    end
    
    toc 
    fprintf("Finish generating eigenvalues.\n")


    %% Generating signal and DFT operator
    if Prior_x == "GB"
        x = randn(N,1);    %高斯信道 
        pos = rand(N,1) < rho;                       
        x= pos.*x;   
    %     x = x./sqrt(rho);
        x = x./norm(x)*sqrt(N);
    elseif Prior_x == "Rad"
        x = sign(randn(N,1));
    end
    Phase_N1 = sign(randn(N,1));
    Phase_N2 = sign(randn(N,1));
    Phase_N3 = sign(randn(N,1));
    
    U_N = @(x) Phase_N3 .* dct( Phase_N2 .* idct( Phase_N1 .* x ) );
    U_Nt = @(x) conj(Phase_N1) .* dct( conj(Phase_N2) .* idct( conj(Phase_N3) .* x ) );

    Y = @(r) theta/N*(x'*r)*x + U_N( d.*U_Nt(r) );

    %% se function
    if Eig_dist=="quartic"
        J_Y = @(Y) gamma*theta*Y-kappa*theta^2*Y.^2+kappa*theta*Y.^3;
        J_lambda1 = theta^2*a2*(gamma+2*a2*kappa)^2 + 1;
        P_func = @(epsilon,Y) 1./(1./epsilon + J_lambda1 - J_Y(Y)); 
        Phi_snr = @(epsilon) 1./integral(@(lam) P_func(epsilon,lam).*rhofun(lam),-2*sqrt(a2),2*sqrt(a2)) - (1/epsilon+1);
    elseif Eig_dist=="sestic"
        J_Y = @(Y) gamma*theta*Y-kappa*theta^2*Y.^2+kappa*theta*Y.^3;
        J_lambda1 = 27/50*Theta^2 + 1;
        P_func = @(epsilon,Y) 1./(1./epsilon + J_lambda1 - J_Y(Y)); 
        Phi_snr = @(epsilon) 1./integral(@(lam) P_func(epsilon,lam).*rhofun(lam),-2*sqrt(a2),2*sqrt(a2)) - (1/epsilon+1);
    
    end
    J_lambda1 = theta^2*a2*(gamma+2*a2*kappa)^2 + 1;
    
%     %% PCAinit  
%     if PCA_init == "T"
%         [max_eig_cal,overlap_cal] = Max_eig(theta,gamma,Eig_dist,d);
%         sigma2_0 = 1/overlap_cal-1;
%         fprintf("sigma2_0 = %e \n", sigma2_0)
%     end
    %% State evolution
    N_SE = 2e7;
    
    sigma2_se = zeros(Iteration,1);
    
    if Prior_x == "GB"
        x_SE = randn(N_SE,1);    %高斯信道 
        pos = rand(N_SE,1) < rho;                       
        x_SE= pos.*x_SE;   
        x_SE = x_SE./norm(x_SE)*sqrt(N_SE);
    elseif Prior_x == "Rad"
        x_SE = sign(randn(N_SE,1));
    end
    
    sigma2_SE = (1-omega)/omega;
    MSE_cal_R = zeros(Iteration,itest);

    if Eig_dist ~= "beta"
        for it = 1:Iteration
        
            % MMSE function
            if Prior_x == "GB"
                r_hat_SE = x_SE + sqrt(sigma2_SE) * randn(N_SE,1);
                [x_hat_post_SE,Var] = MMSE_GB(r_hat_SE,sigma2_SE,rho);
                MMSE = mean(Var);
    %             div = r_hat'*x_hat_post/N/tau2;
            elseif Prior_x == "Rad"
                MMSE = 1 - integral(@(x) normpdf(x).*tanh(1/sigma2_SE-1/sqrt(sigma2_SE)*x), -inf,inf );
            end
            
            dmmse = 1/( 1/MMSE - 1/sigma2_SE );
    
            epsilon = 1/(1-dmmse)-1;
    
            sigma2_SE = 1/Phi_snr(epsilon);
    
            sigma2_se(it) = sigma2_SE;
        
            MSE_cal_R(it,itest) = (1-(1-MMSE)^2)/2;
        
        end
        fprintf("Lambda:%d, MSE_se:%e \n",theta,MSE_cal_R(it,itest))

    end
    
    
    %% Polynomial SE
    
    sigma2_se = zeros(Iteration,1);
    sigma2_SE = (1-omega)/omega;

    for it = 1:Iteration
        
        % MMSE function
        if Prior_x == "GB"
            r_hat_SE = x_SE + sqrt(sigma2_SE) * randn(N_SE,1);
            [x_hat_post_SE,Var] = MMSE_GB(r_hat_SE,sigma2_SE,rho);
            MMSE = mean(Var);
%             div = r_hat'*x_hat_post/N/tau2;
        elseif Prior_x == "Rad"
            MMSE = 1 - integral(@(x) normpdf(x).*tanh(1/sigma2_SE-1/sqrt(sigma2_SE)*x), -inf,inf );
        end
        
        dmmse = 1/( 1/MMSE - 1/sigma2_SE );
        
        epsilon = 1/(1-dmmse)-1;
        
        [~,snr_SE] = Poly_se(d,theta,epsilon,D);
        sigma2_SE = 1/snr_SE;

        sigma2_se(it) = sigma2_SE;
    
        MSE_cal_P(it,itest) = (1-(1-MMSE)^2)/2;
    
    end
    
    fprintf("Lambda:%d, MSE_se:%e \n",theta,MSE_cal_P(it,itest))



    %% Polynomial use se

    r_hat_init = x + sqrt((1-omega)/omega) * randn(N,1);
    r_hat = r_hat_init;
    sigma2 = (1-omega)/omega;
     
    for it = 1:Iteration
    
        % MMSE function
        if Prior_x == "GB"
            [x_hat_post,Var] = MMSE_GB(r_hat,sigma2,rho);
            div = mean(Var)/sigma2;
%             div = r_hat'*x_hat_post/N/tau2;
        elseif Prior_x == "Rad"
            x_hat_post = tanh(r_hat /sigma2);
            div = ( 1 - mean(x_hat_post.^2) ) /sigma2;
            Var = 1 - mean(x_hat_post.^2);
        end
        C = sigma2 / (sigma2 - div*sigma2 );
        x_hat = C*(x_hat_post - div * r_hat);

        dmmse = 1/( 1/mean(Var) - 1/sigma2 );
        epsilon = 1/(1-dmmse)-1;
        
        fprintf("epsilon: test=%e, cal=%e \n",epsilon,norm(x_hat/(x'*x_hat/N) - x)^2/N)
        
        [Alpha,snr] = Poly_se(d,theta,epsilon,D);

        R = zeros(N,D);
        mean_d = zeros(D,1);
        R(:,1) = Y(x_hat);
        mean_d(1) = mean(d);
        for ii = 2:D
            R(:,ii) = Y(R(:,ii-1));
            mean_d(ii) = mean(d.^ii);
        end
        R = R - x_hat*mean_d';
        
        r_hat = R*Alpha;
        
        % scaling and measure
%         alpha = r_hat'*x_hat/N/(1-dmmse);
%         r_hat = r_hat/alpha;
%         sigma2 = norm(r_hat)^2/N-1;
        r_hat = r_hat./sqrt(norm(r_hat)^2/N)*sqrt(1+1/snr);
        sigma2 = 1/snr;

        fprintf("alpha: test=%e, cal=%e \n",1,r_hat'*x/N)
        fprintf("sigma: test=%e, cal=%e, 1/snr=%e \n",sigma2,norm(r_hat./(r_hat'*x/N) - x)^2/norm(x)^2,1/snr)
        
        MSE_sim(it,itest) = MSE(x_hat_post,x,N);

    end
    
end

MSE_sim = mean(MSE_sim,2);
MSE_cal_P = mean(MSE_cal_P,2);
MSE_cal_R = mean(MSE_cal_R,2);


semilogy(1:Iteration,mean(MSE_sim,2),'r-','LineWidth',2)
hold on 
semilogy(1:Iteration,mean(MSE_cal_P,2),'bo','LineWidth',2)
semilogy(1:Iteration,mean(MSE_cal_R,2),'cx','LineWidth',2)


xlabel("Iteration")
ylabel("MSE")
legend("OAMP","SE-Polynomial","SE-Resolvent")




function [hat_x,var_x]=MMSE_GB(R,Sigma,rho)
sigma_x=1/rho;

%% Perform MMSE estimator
Gaussian=@(x,a,A) 1./sqrt(2*pi*A).*exp(-1/2./A.*abs(x-a).^2);
C=(rho.*Gaussian(0,R,Sigma+sigma_x))./...
    ((1-rho).*Gaussian(0,R,Sigma)+rho.*Gaussian(0,R,Sigma+sigma_x));

hat_x=C.*(R*sigma_x)./(sigma_x+Sigma);
var_x=C.*(abs((R*sigma_x)./(sigma_x+Sigma)).^2+(sigma_x*Sigma)./(sigma_x+Sigma))-abs(hat_x).^2;
end
