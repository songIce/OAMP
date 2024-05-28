clear;
close all;
clc;

%This file computes the free cumulants of the spectral density of the quartic ensemble

tic

u = 0; %\mu in the paper

gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27; %\gamma in the paper, strength of the quartic term in the potential

%regularization to avoid dividing by 0
if u == 1
    a2 = 1;
else
    a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
end

% De-comment below to plot the density:
%discretization of the spectral density for the quartic potential
%step = 10^(-6);
%x = -2*sqrt(a2):step:2*sqrt(a2);
%rho = (u+gamma*(2*a2+x.^2)).*sqrt(4*a2-x.^2)/(2*pi);


% figure;
% plot(x, rho)


nmax = 121; % number of free cumulants to find
m = zeros(1, nmax); % allocation of memory for the standard moments of the distribution
m(2) = 1; % first moment is zero by parity, second moment is 1 because of the choices of \mu and \gamma

for i = 4 : 2 : nmax %leaves odd moments =0, sets even moments to their value
    fun = @(x) x.^i .* (u+gamma*(2*a2+x.^2)).*sqrt(4*a2-x.^2)/(2*pi);
    m(i) = integral(fun,-2*sqrt(a2),2*sqrt(a2));
end

% create an auxiliary (nmax X nmax) matrix of coefficients 
A = sym('A',[nmax nmax]);

syms x


% hereby we implement a recursive relation for the free cumulants, see
% Appendix A in arXiv:2106.02356v2. The free cumulants can be obtained from the moments and from the coefficients 
% of a suitable polynomialn (see (A.2) in the mentioned paper) that we are going to build below. 

for j = 1 : nmax 
    
    fprintf('%d\n', j);

    M = x;
    
    % Define recursively a polynomial
    for i = 1 : nmax +1 - j
        M = M + m(i)*x^(i+1);
    end
    % and take its j-th power
    P = (M)^j;

    c = coeffs(P, 'All');
    
    for i = 1 : nmax
        A(j, i) = c(length(c)+1-i-1);
    end
    
end




k = sym('k',[nmax 1]);

k(1) = m(1);

for i = 2 : nmax
    k(i) = 0;
end
    
for i = 2 : nmax
    k(i) = m(i);
    
    for j = 1 : i-1
        k(i) = k(i) - k(j) * A(j, i);
    end
    
    
end

kdouble = double(k);

disp(kdouble);

save freecum_u0.mat k kdouble;

toc




