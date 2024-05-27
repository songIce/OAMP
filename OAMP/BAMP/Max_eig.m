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