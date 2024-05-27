clear;
close all;
clc;

%In this script we implement the Bayes-optimal AMP (BAMP) for the power six
%potential. The algorithm is very similar to the standard AMP with
%rotationally invariant noise, with the observations matrix replaced by its
%pre-processed version. In order to simplify and speed up things a little
%bit we import the Onsagers and other parameters directly from the state
%evolution. Hence this code must be executed only after the state evolution
%has been run (see the file loaded). We suggest reading
%ZF_AMP_single_sestic.m first.

tic

rng(78232);

alphagrid = 2 : 0.2 : 3;
alphagrid = 2.2;
PCA_init = "F";
prior = "sparse";
alpha1 = 1/2; % for 2Points
rho = 0.1; % for sparse
omega = 0.5; % correlation of initialization


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
else
    epsl = 0;
end

n = 2e6;


ntrials = 1;
niter = 10;



if prior == "sparse" || prior == "Rad"
    Generate_Sestic_Onsager(alphagrid,epsl,niter,PCA_init);
elseif prior == "2Points"
    Generate_Sestic_Onsager_2Points(alphagrid,epsl,niter,PCA_init,alpha1);
end
load SE_sestic_Onsager.mat;



max_it = niter; %maximum number of iterations 

%parameters of the sepctral density fo the power six potential 
xi=27/80;
a2=2/3;

tol = 1e-8;

scal_all = zeros(niter, ntrials, length(alphagrid));
overlap = zeros(niter, ntrials, length(alphagrid));
MSE = zeros(niter, ntrials, length(alphagrid));

for j = 1 : length(alphagrid)
    alpha = alphagrid(j);
    %% correlation of initialization
    if PCA_init == "T"
        [~,PCA_overlap] = Max_eig(alpha,0,xi); % PCA results
        epsl = sqrt(PCA_overlap);
        fprintf("epsl:%e \n",epsl);
    end

    SigmaMATall = zeros(niter, niter, ntrials);
    DeltaMATall = zeros(5*niter, 5*niter, ntrials);
    muSEexpall = zeros(niter, ntrials);
    muSEall = zeros(niter, ntrials);
    sigmaexpall = zeros(niter, ntrials);
    tildemuexpall = zeros(5*niter, ntrials);
  
    for i = 1 : ntrials
        
        fprintf('\nalpha=%f, trial #%d\n', alpha, i);
        
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
        x = sqrt(n)*x/norm(x);
        
        Phase_N1 = sign(randn(n,1));
        Phase_N2 = sign(randn(n,1));
        Phase_N3 = sign(randn(n,1));

        U_N = @(x) Phase_N3 .* dct( Phase_N2 .* idct( Phase_N1 .* x ) );
        U_Nt = @(x) conj(Phase_N1) .* dct( conj(Phase_N2) .* idct( conj(Phase_N3) .* x ) );
        

        %generating eigenvalues fo the noise
        eigdiag = zeros(1, n);
        rhofun = @(y) xi.*(6*a2^2+2*a2*y.^2+y.^4).*sqrt(4*a2-y.^2)/(2*pi);
        d = slicesample(1,n,"pdf",rhofun);
 
        %observation
        Y = @(r) alpha/n*(r'*x).*x + U_N( d.*U_Nt(r) );
        JY = @(r) -xi*alpha^2.*Y(Y(r)) - xi*alpha^2.*Y(Y(Y(Y(r)))) + xi*alpha.*Y(Y(Y(Y(Y(r))))); 

        % initialization 
        v0 = epsl * x + sqrt(1-epsl^2) * randn(n, 1);
        normv0 = sqrt(sum(v0.^2));
        u_init = sqrt(n) * v0/normv0;
        
        uAMP = zeros(n, niter+1);
        fAMP = zeros(n, niter);
        tildemuexp = zeros(5*niter, 1);
        tildesigmaexp = zeros(5*niter, 1);
        DeltaMAT = zeros(5*niter, 5*niter);
        Phi = zeros(niter+1, niter+1);
        scal = zeros(niter, 1);
        uAMP(:, 1) = u_init;
        scal_all(1, i, j) = (sum(uAMP(:, 1).* x))^2/sum(x.^2)/sum(uAMP(:, 1).^2);
        
        
        MSE(1, i, j) = 1/n * sum((x-epsl*uAMP(:, 1)).^2);
        fprintf('Iteration %d, MSE=%f, overlap=%f\n', 1, MSE(1, i, j), scal_all(1, i, j));

        for j1 = 1 : niter-1 %iterations
            fAMP(:, j1) = JY(uAMP(:, j1)) - uAMP(:, 1:j1) * Onsager_correct(j1, 1:j1,j)';%local fields
%             uAMP(:, j1+1) = tanh(muSE(j1,j)/sigma2SE(j1,j)*fAMP(:, j1)); %AMP iterate
            uAMP(:, j1+1) = MMSE_func(muSE(j1,j)/sigma2SE(j1,j)*fAMP(:, j1),muSE(j1,j).^2/sigma2SE(j1,j));

            scal_all(j1+1, i, j) = (sum(uAMP(:, j1+1).* x))^2/sum(x.^2)/sum(uAMP(:, j1+1).^2);   %rescaled overlap
            MSE(j1+1, i, j) = sum((uAMP(:, j1+1) - x).^2)/n; %MSE
            fprintf('Iteration %d, MSE=%f, overlap=%f\n', j1+1, MSE(j1+1, i, j), scal_all(j1+1, i, j));
            
            if isnan(MSE(j1+1, i, j))
                if abs(MSE(j1, i, j) - 0.5) < 1e-2
                    MSE(j1+1:niter, i, j) = 0.5;
                    scal_all(j1+1:niter, i, j) = 0; 
                else
                    MSE(j1+1:niter, i, j) = MSE(j1, i, j);
                    scal_all(j1+1:niter, i, j) = scal_all(j1, i, j);
                end
                break;
            elseif abs(MSE(j1+1, i, j) - MSE(j1, i, j))/MSE(j1, i, j) < 1e-6
                MSE(j1+1:niter, i, j) = MSE(j1+1, i, j);
                scal_all(j1+1:niter, i, j) = scal_all(j1+1, i, j);
                break;
            end

        end    
    end
end


BAMP_alphagrid = alphagrid;
BAMP_epsl = epsl;
MSE = 1 - sqrt(1 - 2*MSE);
BAMP_SE_MSE = 1 - sqrt(1 - 2*BAMP_SE_MSE);
BAMP_overlap = mean(scal_all,2);

figure
plot(BAMP_alphagrid,BAMP_SE_overlap(end,:)); hold on
plot(BAMP_alphagrid,BAMP_overlap(end,:))
legend("se","alg")


figure
semilogy(1:niter,MSE(:,end)); hold on
semilogy(1:niter,BAMP_SE_MSE(:,end))
legend("alg","se")


% if length(alphagrid)>1
%     BAMP_MSE = mean(MSE,2);
%     BAMP_overlap = mean(scal_all,2);
%     save BAMP_sestic_alpha0_3 BAMP_alphagrid BAMP_epsl BAMP_MSE BAMP_overlap;
% else
%     BAMP_MSE = MSE;
%     BAMP_overlap = scal_all;
%     filename = strcat("BAMP_sestic_alpha",num2str(alphagrid),"_epsl",num2str(epsl),".mat");
%     save(filename,"BAMP_MSE","BAMP_SE_MSE","BAMP_overlap")
% end

toc