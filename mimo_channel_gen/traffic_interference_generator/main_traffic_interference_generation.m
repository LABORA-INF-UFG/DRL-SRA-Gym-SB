%WARNING: I am afraid I am messing up the time scales a bit (block, sample,
%etc.)

%generate files with traffic and interference
%AK-TODO the best to keep an idea of traffic rate decoupled from framing
%would be to consider that traffic also arrives at uplink and pilot slots
%too but here I will consider it arrives only during DL slots

%The Python code banana_env.py defines
%        #magic numbers used to normalize or as a threshold
%        self.max_SE_value = 7.8  # from dump_SE_values_on_stdout()
%        self.max_tx_packets = 50000 #to normalize the number of transmitted packets
%        self.packet_size_in_bits = 1024  # if 1024, the number of packets per bit coincides with kbps
%also, look for, to get rid of magic numbers such as 10000 below:
%            # SE is an array, with zeros for non-selected users
%            rate_bps = 10000* SE_this_freq * massiveMIMOSystem.BW

clear *
clf
main_folder = 'exp1/'
mkdir([main_folder 'traffic_interference/'])
output_file_name_prefix = 'traffic_interference';
num_episodes = 200;
num_blocks = 100; %per episode
%tau_p = 30;  %samples dedicated to pilots in coherence block
%tau_c = tau_p + 1; %samples in coherence block (samples per frame)
tau_d = 140;
BW = 100e6; %100 MHz - Communication bandwidth
rate_reduction_factor = 10; %factor to reduce the rate and fit the channel

num_bits_per_packet = 1024; %number of bits per packet from Python

method = 3; %1 is flat rate, 2 is to increase rate and then decrease
            %3 is "residential and business"
fixed_rate = 12000; %if method==1

if 0
  %we have the .mat previously generated
  load([main_folder 'all_locations.mat']) %load pre-computed BSpositions and UEpositions
  [K, L]=size(UEpositions);
  eval(['load ' main_folder 'SEs_all_users1.mat']) %assume first frequency
else
  %we will assume some arbitrary values
  K=5;
  L=3;
  SEs_all_users = ones(K,1); %assume SE=1 to all users
end

SEs_target_users = SEs_all_users(:,1);

%in practice the estimation is made with t_p samples, but we will
%assume 1 samples represents all of them (+1 below)
%num_total_samples = num_blocks * (tau_c - tau_p + 1);
num_total_samples = num_blocks * tau_d;
%num_total_samples=100; %number of channel realizations

middle_sample = round(num_total_samples/2);
lambda_business=[linspace(0.5,1,middle_sample) linspace(1,0.5,num_total_samples-middle_sample)];
lambda_residence = 1-0.8*lambda_business;

%% define Poisson lambdas, as packets per second
%look at def incoming_traffic(self) in Python

if 0
    lambdas_all_users = SEs_all_users * BW/K/1000;
else
    lambdas_all_users = SEs_all_users * BW / num_bits_per_packet / rate_reduction_factor;
end

all_lambdas = zeros(K,num_total_samples);

if method == 2

for u=1:K #K is the number of UE's
    if u<K/2
        all_lambdas(u,:) =  lambdas_all_users(u) * lambda_business;
    else
        all_lambdas(u,:) =  lambdas_all_users(u) * lambda_residence;
    end
end

elseif    method == 3
for u=1:K
    if u<K/2
        all_lambdas(u,:) =  lambdas_all_users(u) * ones(size(lambda_business));
    else
        all_lambdas(u,:) =  lambdas_all_users(u) * ones(size(lambda_business));
    end
end

end

for e=1:num_episodes
    disp(['Processing episode ' num2str(e) ' out of ' num2str(num_episodes)])

    output_file_name = [main_folder 'traffic_interference/' output_file_name_prefix '_e_' num2str(e) '.mat'];
    
    if method == 1 %fixed rate
        num_pckts = fixed_rate *ones(K,num_total_samples);
        
    elseif method == 2 %Poisson traffic
        num_pckts = zeros(K, num_total_samples);
        %lambdas_all_users = BW*SE2(:,1)/K/1000;
        for u=1:K
            for t=1:num_total_samples
                num_pckts(u,t) = poissrnd(all_lambdas(u,t)); %,[1, num_total_samples]);
            end
        end
    elseif method == 3 %residential and business
        num_pckts = zeros(K, num_total_samples);
        %lambdas_all_users = BW*SE2(:,1)/K/1000;
        half_users = round(K/2);
        half_time = round(num_total_samples/2);
        for u=1:half_users
            for t=1:half_time
                num_pckts(u,t) = 1.2*poissrnd(all_lambdas(u,t)); %,[1, num_total_samples]);
            end
            for t=half_time+1:num_total_samples
                num_pckts(u,t) = 0.4*poissrnd(all_lambdas(u,t)); %,[1, num_total_samples]);
            end
        end
        for u=half_users+1:K
            for t=1:half_time
                num_pckts(u,t) = 0.4*poissrnd(all_lambdas(u,t)); %,[1, num_total_samples]);
            end
            for t=half_time+1:num_total_samples
                num_pckts(u,t) = 1.2*poissrnd(all_lambdas(u,t)); %,[1, num_total_samples]);
            end
        end
    else
        error('Method not available')
    end
    
    
    %intercell interference
    interference = zeros(K, num_total_samples);
    %will keep it zero for all
    
    eval(['save -v6 ' output_file_name ' num_pckts interference'])
end

subplot(311)
plot(SEs_target_users * BW / 1e6);
xlabel('user #')
ylabel('Channel throughput (Mbps)')
title(['Average channel throughput (Mbps) ' num2str(mean(SEs_target_users * BW / 1e6))])

subplot(312)
average_incoming_rate = mean(num_pckts'*num_bits_per_packet/1e6);
plot(average_incoming_rate, 'r')
xlabel('user #')
ylabel('Average incoming rate (Mbps)')
title(['Total sum rate (Mbps) ' num2str(sum(average_incoming_rate))])

subplot(313)
%plot(num_pckts(1:2:end,:)'*num_bits_per_packet/1e6)
plot(num_pckts'*num_bits_per_packet/1e6)
xlabel('Number of blocks')
ylabel('Instantaneuos incoming rate (Mbps)')
