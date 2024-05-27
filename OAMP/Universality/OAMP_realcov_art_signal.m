%% Test the universality of OAMP
%  The noise matrix is obtained by subtracting the outliers from the real dataset
%  the spike is artificial
%  I test two versions of OAMP
% Method 3 I use the se results to optimal combine the polynomial
%  SE temporary for k = 1 and k = 2
%  ALL algorithms use PCA initialization

clear all;
clc;
close all;

niter = 10; % test number
Iteration = 10;
% polynomial order
k_grid = [1,2,3,4];
% k_grid = [2,6]; 
PCA_init = "F";
sigma2_0 = 3; 
outlier_num = 8; % the number of outliers need to be substracted

Lambda = 3.5; % depends on data, for 1000G sets 3.0, for Hapmap3 sets 3.5

Prior_x = "GB"; % "GB":Gaussian bernuolli prior, "Rad": Rademacher prior
rho = 0.2; %sparse rate, used for GB prior

%% load data 

% Hapmap3 data
data_matrix_all = readmatrix("./data/Hapmap3/Hapmap3.subset.all.matrix.csv");

% 1000G data
% data_matrix_all = readmatrix("./data/1000G/merge_subset_100k_matrix.csv");


% Normalization 
data_matrix_all = zscore(data_matrix_all);
% ground truth cov matrix
cov_full = data_matrix_all*data_matrix_all'/size(data_matrix_all,2);
% system dimension
N = size(data_matrix_all,1);
M = 3000; % sample number

%Initialization 
MSE_sim = zeros(Iteration,length(k_grid),niter);
MSE_SE = zeros(Iteration,length(k_grid),niter);

for itest = 1:niter
    fprintf("Test:%d \n",itest)
    % select columns
    data_matrix = data_matrix_all(:,randperm(size(data_matrix_all,2),M));

    % Normalization 
    cov = data_matrix*data_matrix'/M;
    cov = cov - cov_full; % sample cov matrix - ground truth, calculate the noise

    [UY,d_noise] = eig(cov);
    % [max_eig,posit] = max(diag(Sigma));
    d_noise = diag(d_noise);
    
    % remove the maximum eigenvalue
    d_noise_sort = sort(d_noise,"descend");
    eig_max = d_noise_sort(outlier_num);  % substract the top 8 outlier
    posit = abs(d_noise)<eig_max;
    d_noise(~posit) = 0;
    mean_d = mean(d_noise);
    d_noise = d_noise - mean_d;
    cov = UY*diag(d_noise)*UY';
    cov = (cov+cov')./2; % keep the matrix symmetry, avoid numerical accuracy issues
    cov = cov./norm(cov,'fro')*sqrt(size(cov,1)); % scale the variance = 1
    
    % eigenvalues
    d = eig(cov);
    max_noise_d = max(d);

    %% choose a suitable lambda and generate signal
    
    % generating signal
    if Prior_x == "GB"
        x = randn(N,1);  
        pos = rand(N,1) < rho;                       
        x= pos.*x;   
%         x = x./sqrt(rho);
        x = x./norm(x)*sqrt(N);
    elseif Prior_x == "Rad"
        x = sign(randn(N,1));
    end
    
    
    %% observation
    Y = Lambda/N.*(x*x') + cov;
    [UY,dY] = eig(Y);
    dY = diag(dY);
    sort_s = sort(dY);
    Lambda1 = sort_s(end);
    if abs(Lambda1-max_noise_d) < 1e-4
        fprintf("System SNR is to low, Please adjust the SNR.")
        break;
    end
    fprintf("Eigenvalue 1=%e, 2=%e \n",Lambda1,sort_s(end-1))
    
    [max_S, posit] = max(dY);
    fprintf("PCA overlap=%e \n", UY(:,posit)'*x/norm(UY(:,posit))/norm(x))
    
    if itest == 1
        figure
        histogram(dY,100,'Normalization','pdf')
        hold on
        plot(max_S,0,"x",'LineWidth',2)
    end
    %% Initialization
    % pre-calculate some results for convenience
    Y_mat = zeros(N,N,max(k_grid));
    d_mat = zeros(max(k_grid),N);
    for i=1:max(k_grid)
        Y_mat(:,:,i) = Y^i;
        d_mat(i,:) = dY.^i - mean(dY.^i);
    end
    d_noise = sort(dY,"descend");
    d_noise = d_noise(2:end);
    
    
    % PCA initialization
    [dY_max,posit] = max(dY);
    r_hat_init = UY(:,posit)*sqrt(N);
    dY_sort = sort(dY,"descend");
    % calculate Lambda and overlap by RMT results
    Lambda_cal = 1/mean(1./(dY_sort(1)-dY_sort(2:end)));
    overlap = 1/Lambda_cal^2/mean(1./(dY_sort(1)-dY_sort(2:end)).^2);
    if PCA_init == "T"
        r_hat_init = r_hat_init./sqrt(overlap);
        sigma2_0 = 1/overlap - 1;
        fprintf("Lambda true=%e, cal=%e \n", Lambda,Lambda_cal)
        fprintf("overlap true=%e, cal=%e \n", r_hat_init'*x/N,1)
        fprintf("sigma2 true=%e, cal=%e \n", norm( r_hat_init/(r_hat_init'*x/N)-x)^2/N,sigma2_0)
    else
        r_hat_init = x + sqrt(sigma2_0)*randn(N,1);
    end
    

    % Initialization for state evolution
    N_SE = 2e7;
    if Prior_x == "GB"
        x_SE = randn(N_SE,1);    %高斯信道 
        pos = rand(N_SE,1) < rho;                       
        x_SE= pos.*x_SE;   
        x_SE = x_SE./sqrt(rho);
%         x_SE = x_SE./norm(x_SE)*sqrt(N_SE);
    elseif Prior_x == "Rad"
        x_SE = sign(randn(N_SE,1));
    end
    r_hat_init_SE = x_SE + sqrt(sigma2_0) * randn(N_SE,1);

    
    for ik = 1:length(k_grid)
        k = k_grid(ik);
        
        %% SE
        r_hat_SE = r_hat_init_SE;
        sigma2_SE = sigma2_0;
        
        for it = 1:Iteration
                
            % MMSE function
            if Prior_x == "GB"
                r_hat_SE = x_SE + sqrt(sigma2_SE) * randn(N_SE,1);
                [x_hat_post_SE,Var] = MMSE_X(r_hat_SE,sigma2_SE,rho);
                MMSE = mean(Var);
        %             div = r_hat'*x_hat_post/N/tau2;
            elseif Prior_x == "Rad"
                MMSE = 1 - integral(@(x) normpdf(x).*tanh(1/sigma2_SE-1/sqrt(sigma2_SE)*x), -inf,inf );
            end
            
            dmmse = 1/( 1/MMSE - 1/sigma2_SE );
            
            epsilon = 1/(1-dmmse)-1;
            
            [~,snr_SE] = Poly_se(dY(2:end),Lambda,epsilon,k);
            sigma2_SE = 1/snr_SE;
        
            MSE_SE(it,ik,itest) = MMSE;
        
        end
        
        
        %% Polynomial,  use se results for algorithm
        r_hat = r_hat_init;
        sigma2 = sigma2_0;
         
        for it = 1:Iteration
        
            % MMSE function
            if Prior_x == "GB"
                [x_hat_post,Var] = MMSE_X(r_hat,sigma2,rho);
                div = mean(Var)/sigma2;
            elseif Prior_x == "Rad"
                x_hat_post = tanh(r_hat /sigma2);
                div = ( 1 - mean(x_hat_post.^2) ) /sigma2;
                Var = 1 - mean(x_hat_post.^2);
            end
            C = sigma2 / (sigma2 - div*sigma2 );
            x_hat = C*(x_hat_post - div * r_hat);
        
            dmmse = 1/( 1/mean(Var) - 1/sigma2 );
            epsilon = 1/(1-dmmse)-1;
                    
            % LMMSE part
            [Alpha,snr] = Poly_se(dY(2:end),Lambda_cal,epsilon,k);
            R = zeros(N,k);
            for ii = 1:k
                R(:,ii) = Y_mat(:,:,ii)*x_hat - mean(dY.^ii)*x_hat;
            end
            r_hat = R*Alpha;
            
            %scaling
            r_hat = r_hat./sqrt(norm(r_hat)^2/N)*sqrt(1+1/snr);
            sigma2 = 1/snr;
            %MSE
            MSE_sim(it,ik,itest) = norm(x*sign(x'*x_hat_post)-x_hat_post)^2/norm(x)^2;
            
        end
    
        fprintf("k=%d, mse SE=%e, Alg=%e \n", k, MSE_SE(Iteration,ik,itest),MSE_sim(Iteration,ik,itest))
    
    end


end

MSE_sim = mean(MSE_sim,3);
MSE_SE = mean(MSE_SE,3);

%% Plot

figure
semilogy(1:Iteration,MSE_sim(:,1),'cs','LineWidth',2); hold on
semilogy(1:Iteration,MSE_SE(:,1),'c-','LineWidth',2)
semilogy(1:Iteration,MSE_sim(:,2),'g+','LineWidth',2)
semilogy(1:Iteration,MSE_SE(:,2),'g-','LineWidth',2)
semilogy(1:Iteration,MSE_sim(:,3),'r*','LineWidth',2)
semilogy(1:Iteration,MSE_SE(:,3),'r-','LineWidth',2)
semilogy(1:Iteration,MSE_sim(:,4),'bo','LineWidth',2)
semilogy(1:Iteration,MSE_SE(:,4),'b-','LineWidth',2)
legend("OAMP(k=1)","SE(k=1)","OAMP(k=3)","SE(k=3)","OAMP(k=7)","SE(k=7)","OAMP(k=9)","SE(k=9)")

xlabel("Iteration"); ylabel("MSE")
grid on 












% for Gaussian Bernuolli prior
function [hat_x,var_x]=MMSE_X(R,Sigma,rho)
sigma_x=1/rho;

%% Perform MMSE estimator
Gaussian=@(x,a,A) 1./sqrt(2*pi*A).*exp(-1/2./A.*abs(x-a).^2);
C=(rho.*Gaussian(0,R,Sigma+sigma_x))./...
    ((1-rho).*Gaussian(0,R,Sigma)+rho.*Gaussian(0,R,Sigma+sigma_x));

hat_x=C.*(R*sigma_x)./(sigma_x+Sigma);
var_x=C.*(abs((R*sigma_x)./(sigma_x+Sigma)).^2+(sigma_x*Sigma)./(sigma_x+Sigma))-abs(hat_x).^2;
end


function [max_eig_cal,overlap_cal] = Max_eig(Lambda,u)
    
    gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;
    if u == 1
        a2 = 1;
    else
        a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
    end
    
    sjt = @(z) 1/2*(u*z+ gamma*z.^3 - ( u + 2*a2*gamma + gamma*z.^2 ).*sqrt(z.^2-4*a2));
    sjt_diff = @(z) 1/2*( u + 3*gamma*z.^2 - 2*gamma*z.*sqrt(z.^2-4*a2) - ...
        z.*( u + 2*a2*gamma + gamma*z.^2 )./sqrt(z.^2-4*a2) );

    x_1 = 0;
    x_2 = 3*Lambda;
    itnum = 1000;
    it = 1;
    while it < itnum
        
        max_eig = (x_2 + x_1)/2;
        Lambda_tmp = 1/sjt(max_eig);
        if Lambda_tmp > Lambda
            x_2 = max_eig;
        else
            x_1 = max_eig;
        end
        it = it + 1;
    end
    max_eig_cal = (x_1+x_2)/2;
    
    overlap_cal = -1/Lambda^2/sjt_diff(max_eig_cal);


end