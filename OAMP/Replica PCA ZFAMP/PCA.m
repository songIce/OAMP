% PCA results, predicted by cauchy transform

clear all;
clc;
close all;


% Input snr, Y=theta/N*x*xT + W
Thetagrid = 0.01 : 0.01 : 3; % for algorithm

% Eigenvalue distribution 
Eig_dist = "quartic"; % "quartic", "sestic"
mu = 0; % quartic parameter, mu=1 is wigner

%% Generating eigenvalue    
if Eig_dist == "quartic"
    gamma = (8-9*mu+sqrt(64-144*mu+108*mu.^2-27*mu.^3))/27;
    if mu == 1
        a2 = 1;
    else
        a2 = (sqrt(mu.^2+12*gamma)-mu)./(6*gamma);
    end
    rhofun = @(y) (mu+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi); 
    sjt = @(z) 1/2*(mu*z+ gamma*z.^3 - ( mu + 2*a2*gamma + gamma*z.^2 ).*sqrt(z.^2-4*a2));
    sjt_diff = @(z) 1/2*( mu + 3*gamma*z.^2 - 2*gamma*z.*sqrt(z.^2-4*a2) - ...
        z.*( mu + 2*a2*gamma + gamma*z.^2 )./sqrt(z.^2-4*a2) );

elseif Eig_dist=="sestic"
    mu = 0;
    gamma = 0;
    xi = 27/80;
    a2 = 2/3;
    rhofun = @(y) (mu+2*a2*gamma+6*a2^2*xi+(gamma+2*a2*xi)*y.^2+xi*y.^4).*sqrt(4*a2-y.^2)/(2*pi);
    sjt = @(z) 1/2*(xi*z.^5 - ( 6*a2^2*xi + 2*a2*xi*z.^2 + xi*z.^4 ).*sqrt(z.^2-4*a2));
    sjt_diff = @(z) 1/2*( 5*xi*z.^4 - (4*a2*xi.*z+4*xi.*z.^3).*sqrt(z.^2-4*a2) - ...
        ( 6*a2^2*xi+2*a2*xi*z.^2+xi*z.^4 ).*z./sqrt(z.^2-4*a2) );
else
    a2 = max(d./2)^2;
    sjt = @(z) mean(1./(z-d));
    sjt_diff = @(z) mean(-1./(z-d).^2);
end
    

%% Find the Phase transition point by cauchy transform


Theta_PT = 1/sjt(2*sqrt(a2)); % Phase transition point
max_eig = zeros(length(Thetagrid),1);
overlap = zeros(length(Thetagrid),1);
for i = 1:length(Thetagrid)

    Theta = Thetagrid(i);
    if Theta>Theta_PT
        %bisection
        x_1 = 0;
        x_2 = 6*Theta;
        itnum = 1000;
        it = 1;
        while it < itnum   
            max_eig = (x_2 + x_1)/2;
            Theta_tmp = 1/sjt(max_eig);
            if Theta_tmp > Theta
                x_2 = max_eig;
            else
                x_1 = max_eig;
            end
            it = it + 1;
        end
        max_eig(i) = (x_1+x_2)/2;
        overlap(i) = -1/Theta^2/sjt_diff(max_eig(i));
        
    else
        max_eig(i) = 2*sqrt(a2);
        overlap(i) = 0;
    end

end


PCA_Thetagrid = Thetagrid;
PCA_overlap = overlap;
PCA_MSE = 1 - overlap;
if Eig_dist == "quartic"
    filename = strcat("PCA_u",num2str(mu),"_alpha0_3.mat");
else
    filename = strcat("PCA_sestic_alpha0_3.mat");
end
% save(filename,"PCA_Thetagrid", "PCA_overlap", "PCA_MSE")

plot(PCA_Thetagrid,PCA_overlap)

