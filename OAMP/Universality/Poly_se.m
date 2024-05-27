function [Alpha,snr] = Poly_se(d,Lambda,sigma2,K)

% This function compute the optimal combing of polynomial estimator
% and output snr and the combining coefficient
% d is the eigenvalues of observation matrix
% Lambda is the system snr
% sigma2 is the input snr
% K is the order of polynomial

% compute the moment sequent, note that m(1) = m_0 = 1
m = zeros(2*K+1,1);
m(1) = 1;
for k = 1:2*K
    m(k+1) = mean(d.^k);
end
% compute the alpha, note that alpha(1) = alpha_0 = 1
alpha = zeros(K+1,1);
alpha(1) = 1;
for k = 1:K
    
    alpha_k = 0;
    for i=1:k
        alpha_k = alpha_k + alpha(k+1-i)*m(i);
    end
    alpha(k+1) = Lambda*alpha_k;

end

% compute the covariance matrix
cov = zeros(K);
for k = 1:K
    for l = 1:K
        cov_kl = 0;
        for i = 1:k
            for j = 1:l
                cov_kl = cov_kl + alpha(k+1-i)*alpha(l+1-j)*(m(i+j+1)-m(i+1)*m(j+1));
            end
        end
        cov_kl = cov_kl + (m(k+l+1)-m(k+1)*m(l+1))*sigma2;
        cov(k,l) = cov_kl;
    end
end

% compute beta
beta = zeros(K,1);
for k = 1:K
    beta_k = 0;
    for i = 1:k
        beta_k = beta_k + alpha(i+1)*m(k+1-i);
    end
    beta(k) = beta_k;
end

snr = beta'*cov^-1*beta;
Alpha = cov^-1*beta./snr;

end
