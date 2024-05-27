clear;
close all;
clc;


%This script implements the AMP designed for rotationally invariant noises
%in https://arxiv.org/pdf/2008.11892.pdf. The algorithm requires to import
%the free cumulants.

tic

% alphagrid = 0.9 : 0.1 : 3;
alphagrid = 2;

epsl = 0.5; % correlation of initialization

Eig_dist = "quartic";

%parameters of the quartic potential spectral density

% quartic
% u = 0;
% gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;
% load freecum_u0.mat;

if Eig_dist == "quartic"
    u = 0;
    gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;
    %regularization to avoid dividing by 0
    if u == 1
        a2 = 1;
    else
        a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
    end
    rhofun = @(y) (u+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi);
    load ../data/freecum_u0.mat;
elseif Eig_dist == "sestic"
    mu = 0;
    gamma = 0;
    xi = 27/80;
    a2 = 2/3;
    rhofun = @(y) (mu+2*a2*gamma+6*a2^2*xi+(gamma+2*a2*xi)*y.^2+xi*y.^4).*sqrt(4*a2-y.^2)/(2*pi);
    load ../data/freecum_sestic.mat
elseif Eig_dist == "wigner"
    u = 1;
    a2 = 1;
    rhofun = @(y) sqrt(4-y.^2)/(2*pi);
    load ../data/freecum_u1.mat;
end



tol = 1e-8;

n = 1e6; %dimension of the signal

ntrials = 1; %number of times AMP is run. Each time we generate independent samples.
niter = 30;

max_it = niter;%maximum iterations



freecum = kdouble'; % free cumulants (starting from the 1st)
    
scal_all = zeros(niter, length(alphagrid), ntrials);
scal_allend = zeros(length(alphagrid), ntrials);
MSE = zeros(niter, ntrials, length(alphagrid));

for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);
    SigmaMATall = zeros(niter, niter, ntrials);
    DeltaMATall = zeros(niter, niter, ntrials);
    muSEexpall = zeros(niter, ntrials);
    muSEall = zeros(niter, ntrials);
    sigmaexpall = zeros(niter, ntrials);
  
    for i = 1 : ntrials
        
        fprintf('alpha=%f, trial #%d\n', alpha, i);
        
        %generating a Rademacher signal
        x = sign(randn(n,1));
        Phase_N1 = sign(randn(n,1));
        Phase_N2 = sign(randn(n,1));
        Phase_N3 = sign(randn(n,1));
        
        %generating eigenvalues
        U_N = @(x) Phase_N3 .* dct( Phase_N2 .* idct( Phase_N1 .* x ) );
        U_Nt = @(x) conj(Phase_N1) .* dct( conj(Phase_N2) .* idct( conj(Phase_N3) .* x ) );
                
        d = slicesample(0,n,"pdf",rhofun);

        x_tilde = U_Nt(x);
        Y = @(r) alpha/n*(x'*r).*x + U_N( d.*U_Nt(r) );
                    
        % initializations--------------------------------------------------
        v0 = epsl * x + sqrt(1-epsl^2) * randn(n, 1);
        normv0 = sqrt(sum(v0.^2));
        u_init = sqrt(n) * v0/normv0;
        
        uAMP = zeros(n, niter+1);
        fAMP = zeros(n, niter);
        muSE = zeros(niter, 1);
        muSEexp = zeros(niter, 1);
        sigmaSE = zeros(niter, 1);
        sigmaexp = zeros(niter, 1);
        SigmaMAT = zeros(niter, niter);
        DeltaMAT = zeros(niter, niter);
        Phi = zeros(niter+1, niter+1);
        scal = zeros(niter, 1);
        
        muSE(1) = alpha * epsl;
        sigmaSE(1) = freecum(2);
        SigmaMAT(1, 1) = sigmaSE(1);
        
        uAMP(:, 1) = u_init;
        scal(1) = (sum(uAMP(:, 1).* x))^2/sum(x.^2)/sum(uAMP(:, 1).^2);
        MSE(1, i, j) = norm(x-uAMP(:,1)*epsl)^2/n;
        fprintf('Iteration %d, scal=%f\n', 1, scal(1));
        
        DeltaMAT(1, 1) = (uAMP(:, 1)' * uAMP(:, 1))/n;
        
        b11 = freecum(1);
%         fAMP(:, 1) = Y * uAMP(:, 1) - b11 * uAMP(:, 1); %local fields
        fAMP(:, 1) = Y(uAMP(:, 1)) - b11 * uAMP(:, 1); %local fields
        muSEexp(1) = sum(fAMP(:, 1).* x)/n; %"experimental" \mu
        sigmaexp(1) = sum((fAMP(:, 1) - sum(fAMP(:, 1).* x)/n * x).^2)/n; %experimental variance
        uAMP(:, 2) = tanh(muSE(1)/sigmaSE(1)*fAMP(:, 1)); %AMP iterate
        MSE(2, i, j) = norm(x-uAMP(:,2))^2/n;
        scal(2) = (sum(uAMP(:, 2).* x))^2/sum(x.^2)/sum(uAMP(:, 2).^2); %rescaled overlap
        
        fprintf('Iteration %d, scal=%f\n', 2, scal(2));
        
        %Completing Phi and Delta.
        Phi(2, 1) = muSE(1)/sigmaSE(1) * ( 1 - mean((tanh(muSE(1)/sigmaSE(1)*fAMP(:, 1))).^2));
        DeltaMAT(1, 2) = (uAMP(:, 2)' * uAMP(:, 1))/n;
        DeltaMAT(2, 2) = (uAMP(:, 2)' * uAMP(:, 2))/n;
        DeltaMAT(2, 1) = DeltaMAT(1, 2);
        
        %rest of the iterations
        for jj = 2 : niter-1
            
           Phired = Phi(1:jj, 1:jj); 
           
           %updating B for Onsagers
           B = zeros(jj, jj);
           
           for ii = 0 : jj-1
               B = B + freecum(ii+1) * Phired^ii;
           end
           
           b = B(jj, 1:jj); %extract Onsagers
           
           fAMP(:, jj) = Y(uAMP(:, jj)) - sum(repmat(b, n, 1) .* uAMP(:, 1:jj), 2); %local fields
           muSEexp(jj) = sum(fAMP(:, jj) .* x)/n; %experimental \mu
           sigmaexp(jj) = sum(fAMP(:, jj) - sum(fAMP(:, jj).* x)/n * x).^2/n; %experimental variance
           
           %updating Sigma, covariance matrix of the noises
           Deltared = DeltaMAT(1:jj, 1:jj);
           Sigmared = zeros(jj, jj);
           for i1 = 0 : 2*(jj-1)
               ThetaMAT = zeros(jj, jj);
               
               for i2 = 0 : i1
                   ThetaMAT = ThetaMAT + Phired^i2 * Deltared * (Phired')^(i1-i2);
               end
               
               Sigmared = Sigmared + freecum(i1+2) * ThetaMAT;
           end
           
           muSE(jj) = sqrt(abs(sum(fAMP(:, jj).^2)/n - Sigmared(jj, jj))); %estimating \mu from data
           uAMP(:, jj+1) = tanh(muSE(jj)/Sigmared(jj, jj)*fAMP(:, jj)); %iterate
           
           %rescaled overlap and MSE, output into a file at the end of the program
           scal(jj+1) = (sum(uAMP(:, jj+1).* x))^2/sum(x.^2)/sum(uAMP(:, jj+1).^2);
           MSE(jj+1, i, j) = norm(x-uAMP(:,jj+1))^2/n;

           %adding row and column to DeltaMAT
           for i1 = 1 : jj+1
               DeltaMAT(i1, jj+1) = (uAMP(:, i1)' * uAMP(:, jj+1))/n;
               DeltaMAT(jj+1, i1) = DeltaMAT(i1, jj+1);
           end

           %updating Phi
           Phi(jj+1, jj) = ( 1 - mean((tanh(muSE(jj) / Sigmared(jj, jj) * fAMP(:, jj))).^2)) * muSE(jj) / Sigmared(jj, jj);
           
           fprintf('Iteration %d, scal=%f\n', jj+1, scal(jj+1));
           
        end
        
        %collecting results
        muSEexpall(:, i)=muSEexp;
        muSEall(:, i)=muSE;
        sigmaexpall(:, i)=sigmaexp;
        
        
        SigmaMATall(1:niter-1, 1:niter-1, i) = Sigmared;
        DeltaMATall(:, :, i) = DeltaMAT;
        
        
        scal_all(:, j, i) = scal;
        scal_allend(j, i) = scal(end);
        
    end
end



toc




ZFAMP_MSE = mean(MSE,2);
ZFAMP_overlap = scal_all;
ZFAMP_alphagrid = alphagrid;

if length(alphagrid) > 1
    if Eig_dist == "sestic"
        filename = strcat("ZFAMP_single_sestic_alpha0_3.mat");
    else
        filename = strcat("ZFAMP_single_u",num2str(u),"_alpha0_3.mat");
    end
else
    if Eig_dist == "sestic"
        filename = strcat("ZFAMP_single_sestic_alpha",num2str(alphagrid),"_epsl",num2str(epsl),".mat");
    else
        filename = strcat("ZFAMP_single_u",num2str(u),"_alpha",num2str(alphagrid),"_epsl",num2str(epsl),".mat");
    end
end
save(filename,"ZFAMP_MSE","ZFAMP_overlap","ZFAMP_alphagrid")