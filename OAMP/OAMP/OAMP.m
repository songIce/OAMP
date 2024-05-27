%% Resolvent approach, optimal v chooses by 1/tau2 + J(lambda1) 
% and comprare with the search one 

clear all;
clc;
close all;

Iteration = 30;
N = 1e6;

PCA_init = "F";
omega = 0;  % initial omega in [0,1]

% Input snr, Y=theta/N*x*xT + W
Thetagrid = 0.1 : 0.1 : 3;
Thetagrid = 2.5;

% Eigenvalue distribution 
Eig_dist = "quartic"; % "quartic", "sestic"
mu = 0; % quartic parameter, mu=1 is wigner distribution

% signal distribution
prior = "2Points"; % "Rad", "sparse", "3Points"
sparsity = 0.02;
alpha1 = 1/2;
alpha2 = 1/5;

%% Generating eigenvalue    
tic
if Eig_dist == "quartic"
    gamma = (8-9*mu+sqrt(64-144*mu+108*mu.^2-27*mu.^3))/27;
    xi = 0;
    %regularization to avoid dividing by 0
    if mu == 1
        a2 = 1;
    else
        a2 = (sqrt(mu.^2+12*gamma)-mu)./(6*gamma);
    end
    rhofun = @(y) (mu+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi); 
elseif Eig_dist == "sestic"
    % In Jean's paper, they consider pure sestic distirbution, u, gamma, xi
    % are fixed
    mu = 0;
    gamma = 0;
    xi = 27/80;
    a2 = 2/3;
    rhofun = @(y) (mu+2*a2*gamma+6*a2^2*xi+(gamma+2*a2*xi)*y.^2+xi*y.^4).*sqrt(4*a2-y.^2)/(2*pi);
end
d = slicesample(1,N,"pdf",rhofun); % Generating eigenvalues 
d = d-mean(d);
toc 
fprintf("Finish generating eigenvalues.\n")

%% MSE
% || xxT - rrT ||_F
MSE = @(x,r,n) 1/n * norm(x-r)^2 ;
MSE_SE = zeros(Iteration,length(Thetagrid));
Overlap_SE = zeros(Iteration,length(Thetagrid));
MSE_sim = zeros(Iteration,length(Thetagrid));
Overlap_sim = zeros(Iteration,length(Thetagrid));
if prior == "sparse"
    MMSE_func = @(r,snr_LE) sqrt(sparsity).*sinh((r)./sqrt(sparsity)).*exp(-snr_LE/(2*sparsity))./(1-sparsity+sparsity.*cosh((r)./sqrt(sparsity)).*exp(-snr_LE/(2*sparsity)));
    aux_func_se=@(X,snr_LE) X*integral(@(Z) normpdf(Z).*sqrt(sparsity).*sinh((Z.*sqrt(snr_LE)+X.*snr_LE)./sqrt(sparsity))*exp(-1/(2*sparsity)*snr_LE)./ ...
        (1-sparsity+sparsity*cosh((Z.*sqrt(snr_LE)+X.*snr_LE)./sqrt(sparsity))*exp(-snr_LE/(2*sparsity))),-16,16);
    MMSE_func_SE = @(snr_LE) 1 - ( sparsity/2*(aux_func_se(-1/sqrt(sparsity),snr_LE)+aux_func_se(1/sqrt(sparsity),snr_LE)) + (1-sparsity)*aux_func_se(0,snr_LE) );
elseif prior == "Rad"
    MMSE_func = @(r,v) tanh(r);
    MMSE_func_SE = @(snr_LE) 1 - integral(@(x) normpdf(x).*tanh(snr_LE-sqrt(snr_LE)*x), -inf,inf ); 
elseif prior =="3Points"
    aux1 = @(X,Z,snr_LE) exp(-snr_LE/2/alpha1^2 + (Z.*sqrt(snr_LE)+X.*snr_LE)./alpha1 );
    aux2 = @(X,Z,snr_LE) exp(-snr_LE/2/alpha2^2 + (Z.*sqrt(snr_LE)+X.*snr_LE)./alpha2 );
    aux=@(X,snr_LE) integral(@(Z) X.*normpdf(Z).*( alpha1/2.*aux1(X,Z,snr_LE) + alpha2/2.*aux2(X,Z,snr_LE) )./( 1 - alpha1^2/2 - alpha2^2/2 + alpha1^2/2.*aux1(X,Z,snr_LE) + alpha2^2/2.*aux2(X,Z,snr_LE) ), -20, 20);
    MMSE_func_SE = @(snr_LE) 1 - (alpha1^2/2.*aux(1/alpha1,snr_LE) + alpha2^2/2.*aux(1/alpha2,snr_LE));
    MMSE_func = @(r,snr_LE) ( alpha1/2.*exp(-snr_LE/2/alpha1^2 + r./alpha1 ) + alpha2/2.*exp(-snr_LE/2/alpha2^2 + r./alpha2 ) )./...
        ( 1 - alpha1^2/2 - alpha2^2/2 + alpha1^2/2.*exp(-snr_LE/2/alpha1^2 + r./alpha1 ) + alpha2^2/2.*exp(-snr_LE/2/alpha2^2 + r./alpha2 )  );
elseif prior =="2Points"

    MMSE_func = @(r,snr_LE) ( alpha1.*exp(-snr_LE/2/alpha1^2 + r./alpha1 ) )./...
        ( 1 - alpha1^2 + alpha1^2.*exp(-snr_LE/2/alpha1^2 + r./alpha1 ) );
    aux1 = @(X,Z,snr_LE) exp(snr_LE/2/alpha1^2 - (Z.*sqrt(snr_LE)+X.*snr_LE)./alpha1 );
    aux=@(X,snr_LE) integral(@(Z) X.*normpdf(Z).*alpha1./( (1 - alpha1^2).*aux1(X,Z,snr_LE) + alpha1^2 ), -Inf, Inf);
    MMSE_func_SE = @(snr_LE) 1 - (alpha1^2.*aux(1/alpha1,snr_LE) + (1-alpha1^2).*aux(0,snr_LE) );

end
%% Generating signal and DFT operator
if prior == "sparse"
    x = sign(randn(N,1));
    x(rand(N,1)>sparsity) = 0;
    x = x./sqrt(sparsity);
elseif prior == "Rad"
    x = sign(randn(N,1));
elseif prior == "3Points"
    x = zeros(N,1);
    nonzeroterm = [ repmat(1/alpha1, round(N*alpha1^2/2),1); repmat(1/alpha2, round(N*alpha2^2/2),1) ];
    x(randperm(N, length(nonzeroterm) )) = nonzeroterm;
elseif prior == "2Points"
    x = zeros(N,1);
    x(randperm(N,round(N*alpha1^2))) = 1/alpha1;
end

Phase_N1 = sign(randn(N,1));
Phase_N2 = sign(randn(N,1));
Phase_N3 = sign(randn(N,1));

U_N = @(x) Phase_N3 .* dct( Phase_N2 .* idct( Phase_N1 .* x ) );
U_Nt = @(x) conj(Phase_N1) .* dct( conj(Phase_N2) .* idct( conj(Phase_N3) .* x ) );

%% Testing
for itest = 1:length(Thetagrid)
    
    Theta = Thetagrid(itest);
    [~,overlapPCA] = Max_eig(Theta,mu,xi);
    if PCA_init =="T"
        omega = overlapPCA;
    end
    fprintf("Theta = %.3f, omega = %.3f \n",Theta,omega)
    
    %% se function
    if Eig_dist=="quartic"
        J_Y = @(Y) mu.*Theta.*Y-gamma.*Theta.^2.*Y.^2+gamma.*Theta.*Y.^3;
        J_lambda1 = Theta.^2.*a2*(mu+2*a2*gamma)^2 + 1;
        Phi_lambda = @(Y) 1./(J_lambda1-J_Y(Y));
    elseif Eig_dist=="sestic"
        J_Y = @(Y) xi.*Theta.*Y.^5-xi.*Theta.^2.*Y.^4-xi.*Theta.^2.*Y.^2;
        J_lambda1 = 27/50*Theta^2 + 1;
        Phi_lambda = @(Y) 1./(J_lambda1-J_Y(Y));
    end

    P_func = @(rho,Y) 1./(rho + J_lambda1 - J_Y(Y)); 
    Phi_snr = @(rho) 1./integral(@(lam) P_func(rho,lam).*rhofun(lam),-2*sqrt(a2),2*sqrt(a2)) - (rho+1);



    %% algorithm function
    
    x_tilde = U_Nt(x);
    X_D = d.*x_tilde;
    X_D2 = d.*X_D;   
    X_D3 = d.*X_D2;
    X_D4 = d.*X_D3;
    P_mean = @(epsilon) mean(P_func(epsilon,d)); 
    if Eig_dist=="quartic"
        P_mat = [ x_tilde, X_D, X_D2 ];
        QT_mat = [ mu.*Theta.^2.*x_tilde + gamma.*Theta.^2.*X_D2, gamma.*Theta.^2.*X_D, gamma.*Theta.^2.*x_tilde]';
        P_function_d = @(x_df,epsilon) x_df.*P_func(epsilon,d) + 1/N.*bsxfun(@times, P_func(epsilon,d), P_mat) * ...
            ( eye(3) - 1/N.*bsxfun(@times, QT_mat, P_func(epsilon,d)') * P_mat)^(-1) * ...
            ( bsxfun(@times, QT_mat, P_func(epsilon,d)') * x_df ) - P_mean(epsilon).*x_df; 
    elseif Eig_dist=="sestic"
        P_mat = [x_tilde, X_D, X_D2, X_D3, X_D4];
        QT_mat = [X_D4 X_D3, X_D2, X_D, x_tilde]';
        P_function_d = @(x_df,epsilon) x_df.*P_func(epsilon,d) + xi.*Theta.^2./N.*bsxfun(@times, P_func(epsilon,d), P_mat) * ...
            ( eye(5) - xi.*Theta.^2./N.*bsxfun(@times, QT_mat, P_func(epsilon,d)') * P_mat)^(-1) * ...
            ( bsxfun(@times, QT_mat, P_func(epsilon,d)') * x_df ) - P_mean(epsilon).*x_df; 
    end
    
    P_function = @(x_df,epsilon) U_N(P_function_d(U_Nt(x_df),epsilon));

    %% State evolution
    v_opt = zeros(Iteration,1);
    snr_opt_SE = zeros(Iteration,1);
    if PCA_init ~= "T" && MMSE_func_SE(0) < 1e-6
        % non-symmetry distribution uses random initialization
        snr_LE_SE = 0;
    else
        snr_LE_SE = omega/(1-omega);
    end
    
    
    for it = 1:Iteration
        
        % MMSE function
        MMSE_SE = MMSE_func_SE(snr_LE_SE);
        MSE_SE(it,itest) = real(MMSE_SE);
        Overlap_SE(it,itest) = abs(1 - real(MMSE_SE));

        dmmse_SE = 1/( 1/MMSE_SE - snr_LE_SE);
        rho_SE = 1/(dmmse_SE) - 1;
        
        % div-free nonlinear step
        snr_LE_SE = Phi_snr(rho_SE);

        v_opt(it) = rho_SE;
        snr_opt_SE(it) = snr_LE_SE;
         
    end
    

    fprintf("Lambda:%d, MSE_se:%e \n",Theta,MSE_SE(it,itest))
    

    %% OAMP
    % Initializing
    if PCA_init ~= "T" && MMSE_func_SE(0) < 1e-6
        % non-symmetry distribution uses random initialization
        r_hat = randn(N,1);
        alpha_sim = 0;
        sigma2_sim = 1;
    else
        r_hat = sqrt(omega) * x + sqrt(1-omega) * randn(N,1);
        alpha_sim = sqrt(omega);
        sigma2_sim = 1-omega;
    end

    for it = 1:Iteration
    
        % MMSE function
        x_hat_post = MMSE_func(r_hat * alpha_sim/sigma2_sim,alpha_sim^2/sigma2_sim);
        div = ( 1 - mean(x_hat_post.^2) ) * alpha_sim/sigma2_sim;
        
        x_hat = x_hat_post - div * r_hat;
        
        epsilon = v_opt(it);

%       Note that in this place we use the result from se to deteremine the
%       parameter v. however, if the system dimension is low, this
%       parameter is incorrect, at this time we should compute the maximum
%       eigenvalue of the real data matrix and compute the sutible v.

        r_hat = P_function(x_hat,epsilon);        


        % scaling
%         alpha = r_hat'*x/norm(x)^2;
%         r_hat = r_hat/alpha;
%         alpha = 1;
%         % measure
%         alpha_sim =  r_hat'*x/norm(x)^2;
%         sigma2_sim = norm(r_hat - alpha_sim * x)^2/norm(x)^2;      

        r_hat = r_hat/(norm(r_hat)/sqrt(N))*sqrt(1+1/snr_opt_SE(it));
        alpha_sim = 1;
        sigma2_sim = 1/snr_opt_SE(it);
        

    
        MSE_sim(it,itest) = MSE(x_hat_post,x,N);
        Overlap_sim(it,itest) = (x_hat_post'*x)^2/norm(x_hat_post)^2/norm(x)^2;

    end
    
    fprintf("Lambda:%d, MSE_sim:%e \n",Thetagrid(itest),MSE_sim(it,itest))

end


if length(Thetagrid)>1

    figure;
    semilogy(Thetagrid,MSE_sim(Iteration,:),'b-',Thetagrid, MSE_SE(Iteration,:),'ro','LineWidth',2);

    xlabel('Lambda');
    ylabel('MSE');
    grid on

    figure;
    plot(Thetagrid,Overlap_sim(Iteration,:),'b-',Thetagrid, Overlap_SE(Iteration,:),'ro','LineWidth',2);
    hold on
    xlabel('Iteration');
    ylabel('overlap');
    legend('OAMP','OAMP-SE');
    
    sjt = @(z) 1/2*(mu*z+ gamma*z.^3 - ( mu + 2*a2*gamma + gamma*z.^2 ).*sqrt(z.^2-4*a2));
    PCA_PT_lam = 1/sjt(2*sqrt(a2));
    
    xline(real(PCA_PT_lam))
    legend('OAMP','OAMP-SE',"PCA Phase transition")


else

    figure;
    semilogy(1:Iteration,MSE_sim,'b-',1:Iteration, MSE_SE,'ro','LineWidth',2);
    hold on
    xlabel('Iteration');
    ylabel('MSE');
    legend('OAMP','OAMP-SE');

    figure;
    plot(1:Iteration,Overlap_sim,'b-',1:Iteration, Overlap_SE,'ro','LineWidth',2);
    hold on
    xlabel('Iteration');
    ylabel('overlap');
    legend('OAMP','OAMP-SE');

end






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