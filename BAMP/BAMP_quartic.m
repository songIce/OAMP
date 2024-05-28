clear;
close all;
clc;

% This script implements the Bayes-optimal AMP (BAMP) for the quartic
% potential based on the paper: https://arxiv.org/pdf/2210.01237.
% The corresponding code is available at: https://github.com/fcamilli95/Structured-PCA.
% 
% The main difference lies in the prior distribution and the initialization.
% We initialize the input of the nonlinear denoiser as 
% r0 = sqrt(omega) * x0 + sqrt(1 - omega) * v0.
% However, BAMP initializes the output of the nonlinear denoiser as 
% x_hat0 = epsl * x0 + sqrt(1 - epsl^2) * v0.
% We do not attempt to change the state evolution and the algorithm; thus, we calculate
% epsl based on the given r0 and use it as the initialization for BAMP.
% The other details refer to https://github.com/fcamilli95/Structured-PCA.

% There are mainly three kinds of prior distributions:
% 1. Rademacher
% 2. Sparse: the parameter rho controls the sparsity
% 3. 2-Point prior: alpha1^2 * delta(1/alpha1) + (1 - alpha1^2) * delta(0)

% PCA_init: This is not a realistic PCA initialization. We only calculate
% the overlap results of PCA by random matrix results and use these results
% as the initialization omega/epsl.

tic


alphagrid = 0.6 : 0.1 : 2;
alphagrid = 2;
PCA_init = "F";
prior = "sparse"; 
rho = 0.1; % for sparse
alpha1 = 1/3; % for 2Points
omega = 0; % correlation of initialization
n = 2e6; % system dimension
ntrials = 1; 
niter = 16;

if prior == "sparse"

    MMSE_func = @(r,snr_LE) sqrt(rho).*sinh((r)./sqrt(rho))*exp(-snr_LE/(2*rho))./(1-rho+rho*cosh((r)./sqrt(rho))*exp(-snr_LE/(2*rho)));
    aux_func=@(X,snr_LE) X*integral(@(Z) normpdf(Z).*sqrt(rho).*sinh((Z.*sqrt(snr_LE)+X.*snr_LE)./sqrt(rho))*exp(-1/(2*rho)*snr_LE)./(1-rho+rho*cosh((Z.*sqrt(snr_LE)+X.*snr_LE)./sqrt(rho))*exp(-snr_LE/(2*rho))),-16,16);
    MMSE_func_SE = @(snr_LE) 1 - ( rho/2*(aux_func(-1/sqrt(rho),snr_LE)+aux_func(1/sqrt(rho),snr_LE)) + (1-rho)*aux_func(0,snr_LE) );
elseif prior == "Rad"
    rho = 1;
    MMSE_func = @(r,v) tanh(r);
    MMSE_func_SE = @(snr_LE) 1 - integral(@(x) normpdf(x).*tanh(snr_LE-sqrt(snr_LE)*x), -inf,inf ); 
elseif prior =="2Points"

    MMSE_func = @(r,snr_LE) ( alpha1.*exp(-snr_LE./2./alpha1.^2 + r./alpha1 ) )./...
        ( 1 - alpha1.^2 + alpha1.^2.*exp(-snr_LE./2./alpha1.^2 + r./alpha1 ) );
    aux1 = @(X,Z,snr_LE) exp(snr_LE./2./alpha1.^2 - (Z.*sqrt(snr_LE)+X.*snr_LE)./alpha1 );
    aux=@(X,snr_LE) integral(@(Z) X.*normpdf(Z).*alpha1./( (1 - alpha1.^2).*aux1(X,Z,snr_LE) + alpha1.^2 ), -Inf, Inf);
    MMSE_func_SE = @(snr_LE) 1 - (alpha1^2.*aux(1/alpha1,snr_LE) + (1-alpha1^2).*aux(0,snr_LE) );

end

% The code of BAMP initialize the input of LE estimator, to keep fair
% comparation with OAMP, we initialize the input of denoiser as
% r_hat = sqrt(omega)*x + sqrt(1-sqrt(omega)), then compute the output 
% rho = mmse(omega) and detemine the initial epsl in BAMP
% epsl = sqrt(1-rho)
if PCA_init ~= "T"
    rho_omega = MMSE_func_SE(omega/(1-omega));
    epsl = sqrt(1-rho_omega);
    fprintf("epsl:%e \n",epsl);
    if epsl==0 % for symmetry prior, we need some initialization
        epsl = 0.5;
    end
else
    epsl = 0;
end



max_it = niter; %maximum number of iterations, it matches that of SE. 

%parameters of the quartic potential spectral density
u = 0;
gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;

%regularization to avoid dividing by 0
if u == 1
    a2 = 1;
else
    a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
end

if prior == "sparse" || prior == "Rad"
    Generate_Quartic_Onsager(u,alphagrid,epsl,niter,rho,PCA_init);
elseif prior == "2Points"
    Generate_Quartic_Onsager_2Points(u,alphagrid,epsl,niter,alpha1,PCA_init);
end
load SE_quartic_Onsager.mat;


tol = 1e-8;

scal_all = zeros(niter, ntrials, length(alphagrid));
overlap = zeros(niter, ntrials, length(alphagrid));
MSE = zeros(niter, ntrials, length(alphagrid));

for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);
    %% correlation of initialization
    if PCA_init == "T"
        [~,PCA_overlap] = Max_eig(alpha,u,0); % PCA results
        epsl = sqrt(PCA_overlap);
        fprintf("epsl:%e \n",epsl);
    end


    SigmaMATall = zeros(niter, niter, ntrials);
    DeltaMATall = zeros(3*niter, 3*niter, ntrials);
    muSEexpall = zeros(niter, ntrials);
    muSEall = zeros(niter, ntrials);
    sigmaexpall = zeros(niter, ntrials);
    tildemuexpall = zeros(3*niter, ntrials);
    
    for i = 1 : ntrials
        
        fprintf('alpha=%f, trial #%d\n', alpha, i);

        %coefficients of the preprocessing polynomial
        c1 = u * alpha;
        c2 = -gamma * alpha^2;
        c3 = gamma * alpha;
        
        %generating signal
        if prior == "sparse"
            x = sign(randn(n,1));
            x(rand(n,1)>rho) = 0;
            x = x./sqrt(rho);
        elseif prior == "Rad"
            x = sign(randn(n,1));
        elseif prior == "2Points"
            x = zeros(n,1);
            x(randperm(n,round(n*alpha1^2))) = 1/alpha1;
        end

        Phase_N1 = sign(randn(n,1));
        Phase_N2 = sign(randn(n,1));
        Phase_N3 = sign(randn(n,1));

        U_N = @(x) Phase_N3 .* dct( Phase_N2 .* idct( Phase_N1 .* x ) );
        U_Nt = @(x) conj(Phase_N1) .* dct( conj(Phase_N2) .* idct( conj(Phase_N3) .* x ) );

        %generating eigenvalues fo the noise
        eigdiag = zeros(1, n);
        rhofun = @(y) (u+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi);
        d = slicesample(rand(1),n,"pdf",rhofun);
 
        %observation
        Y = @(r) alpha/n*(r'*x).*x + U_N( d.*U_Nt(r) );
        JY = @(r) u*alpha.*Y(r) - gamma*alpha^2.*Y(Y(r)) + gamma*alpha.*Y(Y(Y(r)));
                
        % initialization 
        v0 = epsl * x + sqrt(1-epsl^2) * randn(n, 1);
        normv0 = sqrt(sum(v0.^2));
        u_init = sqrt(n) * v0/normv0;
        
        uAMP = zeros(n, niter+1);
        fAMP = zeros(n, niter);
        tildemuexp = zeros(3*niter, 1);
        tildesigmaexp = zeros(3*niter, 1);
        DeltaMAT = zeros(3*niter, 3*niter);
        Phi = zeros(niter+1, niter+1);
        scal = zeros(niter, 1);
        uAMP(:, 1) = u_init;

        scal_all(1, i, j) = (sum(uAMP(:, 1).* x))^2/sum(x.^2)/sum(uAMP(:, 1).^2);
        
        MSE(1, i, j) = 1/n * (norm(epsl*uAMP(:, 1) - x))^2 ;
        fprintf('Iteration %d, MSE=%f, overlap=%f\n', 1, MSE(1, i, j), scal_all(1, i, j));


        for j1 = 1 : niter-1 %iterations
            
            fAMP(:, j1) = JY(uAMP(:, j1)) - uAMP(:, 1:j1) * Onsager_correct(j1, 1:j1,j)'; %local field

%             uAMP(:, j1+1) = tanh(muSE(j1,j)/sigma2SE(j1,j)*fAMP(:, j1)); %AMP iterate
            uAMP(:, j1+1) = MMSE_func(muSE(j1,j)/sigma2SE(j1,j)*fAMP(:, j1),muSE(j1,j).^2/sigma2SE(j1,j));
            
            scal_all(j1+1, i, j) = (sum(uAMP(:, j1+1).* x))^2/sum(x.^2)/sum(uAMP(:, j1+1).^2); %rescaled overlap
            MSE(j1+1, i, j) = 1/n * (norm(uAMP(:, j1+1) - x))^2; %MSE
            
            fprintf('Iteration %d, MSE=%f, overlap=%f\n', j1+1, MSE(j1+1, i, j), scal_all(j1+1, i, j));
            

            if isnan(MSE(j1+1, i, j)) 
                MSE(j1+1:niter, i, j) = MSE(j1, i, j);
                scal_all(j1+1:niter, i, j) = scal_all(j1, i, j);
                break;
            elseif abs(MSE(j1+1, i, j) - MSE(j1, i, j))/MSE(j1, i, j) < 1e-5
                MSE(j1+2:niter, i, j) = MSE(j1+1, i, j);
                scal_all(j1+2:niter, i, j) = scal_all(j1+1, i, j);
                break;
            end
        
        end   
        
    end
end

toc


BAMP_alphagrid = alphagrid;
BAMP_overlap = mean(scal_all,2);
BAMP_epsl = epsl;
BAMP_u = u;

% if length(alphagrid)>1
%     BAMP_MSE = mean(MSE,2);
%     BAMP_overlap = mean(scal_all,2);
%     filename = strcat("BAMP_u",num2str(u),"_alpha0_3.mat");
%     save(filename,"BAMP_alphagrid", "BAMP_epsl", "BAMP_MSE", "BAMP_u", "BAMP_overlap")
% else
%     
%     BAMP_MSE = MSE;
%     BAMP_overlap = scal_all;
%     filename = strcat("BAMP_u",num2str(u),"_alpha",num2str(alphagrid),"_epsl",num2str(epsl),".mat");
%     save(filename,"BAMP_MSE","BAMP_SE_MSE", "BAMP_overlap")
% end

figure
plot(BAMP_alphagrid,BAMP_SE_overlap(end,:)); hold on
plot(BAMP_alphagrid,BAMP_overlap(end,:))
legend("se","alg")


figure
semilogy(1:niter,MSE(:,end),'ro'); hold on
semilogy(1:niter,BAMP_SE_MSE(:,end),'b-')
legend("alg","se")


