function evaluate_correlation_with_estimated_channel(R,H,Hhat,target_cell_index)
%AK this function look at correlation between h[n] and h[n-1]
%see also evaluate_correlation_with_estimated_channel
%R is 4D because it includes data for all cells, but H and Hhat are 3D
%because it concerns only the target cell.

L=size(R,4); %number of base station (BS)
K=size(R,3); %number of UE per BS
%M=size(R,1); %number of antennas at BS (number at UE is always 1)

%mean_corr_coef_over_time = zeros(K,L);
mean_corr_coef_over_time = zeros(K,1);

for u=1:K
    for b=target_cell_index %only target cell %for b=1:L
        clf
        R_ue = squeeze(R(:,:,u,b)); %size(R) = 64    64    10     7
        
        %issymmetric(R_ue) %true
        %issymmetric(M) %false
        [~,p] = chol(R_ue);
        if p > 0
            warning('AK: Non positive definite')
            disp(['b = ' num2str(b) ' u= ' num2str(u)])
        end
        if isreal(R_ue)
            disp(['b = ' num2str(b) ' u= ' num2str(u) ' => R is real-valued'])
        else
            disp(['b = ' num2str(b) ' u= ' num2str(u) ' => R is complex-valued'])
        end
        
        HOverTime = squeeze(H(:,:,u));
        HhatOverTime = squeeze(Hhat(:,:,u));
        
        [M,num_realizations]=size(HOverTime); %64   100
        [M,num_blocks]=size(HhatOverTime); %64
        %with respecto to target BS
        correlation_coeffs = zeros(num_realizations-1,1);
        tau_d = num_realizations / num_blocks; %e.g. 140
        
        r=1; 
        for num_b=1:num_blocks
            thisHhat = HhatOverTime(:,num_b);
            for i=1:tau_d
                thisH = HOverTime(:,r);                
                correlation_coeffs(r)=abs(dot(thisH,thisHhat))/(norm(thisH)*norm(thisHhat));
                r = r + 1; %update realization counter
            end
        end
        
        mean_corr_coef_over_time(u) = mean(correlation_coeffs);
        plot(correlation_coeffs);
        hold on
        plot(1:tau_d:length(correlation_coeffs),correlation_coeffs(1:tau_d:end),'x','MarkerSize',18);
        title(['b = ' num2str(b) ' u= ' num2str(u)])
        ylabel('Correlation coefficient between H[n] and Hhat[m]')
        pause %(1.8)
        disp('finished user')
    end
end
clf
%imagesc(mean_corr_coef_over_time), colorbar
plot(mean_corr_coef_over_time)
title('Average correlation coefficient between H[n] and Hhat[m]')
xlabel('UE'), ylabel('Average correlation')