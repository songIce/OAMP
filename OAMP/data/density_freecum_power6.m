clear;
close all;
clc;

%The following script find the free cumulants for the pure power six
%potential

tic

% set the parameters of the potential, so that the spectral density has
% unit variance.
a2=2./3.;
xi=27./80.;


nmax = 120; %number of free cumulants needed
m = zeros(1, nmax); % allocate memory for the moments
m(2) = 1; %second moment equals one thanks to the choice of a2 and xi.

for i = 4 : 2 : nmax % leaves the odd moments equal to 0 and sets the others to their value using the spectral density
    fun = @(x) x.^i .* xi.*(6*a2^2+2*a2*x.^2+x.^4).*sqrt(4*a2-x.^2)/(2*pi);
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
disp(kdouble)
save freecum_sestic.mat k kdouble;

toc




