function correlation_coeffs = ak_ita2019_calculate_correlation_coeff(H)

[M,num_realizations,K,L]=size(H); %64   100    10     7
%with respecto to target BS
 correlation_coeffs = NaN*ones(num_realizations,K,K,L-1);

angle_this = [];
for r=1:num_realizations
    for b=2:L %from 2nd BS
        for u_target=1:K
            H_target = H(:,r,u_target,1); %1st BS is the target
            %u_target
            for u=1:K
                H_u = H(:,r,u,b);
                %u
                correlation_coeffs(r,u_target,u,b)=abs(dot(H_target,H_u))/(norm(H_target)*norm(H_u));
                angle_this = [angle_this acos(abs(dot(H_target,H_u))/(norm(H_target)*norm(H_u)))*180/pi];
            end
        end
    end
end
if 0
    close all
    plot(angle_this)
    pause
end