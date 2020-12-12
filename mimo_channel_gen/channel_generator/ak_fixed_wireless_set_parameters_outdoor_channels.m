% From:
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   Set the values for the main PAR for subsequent simulations.
%
%   NOTE: to initialize PAR, either use scenario_idx = 0 or call the
%   function without any argument.
%   
%   Input
%       -   scenario_idx: index of the simulation scenario
%   
%   Output:
%       -   PAR: a structure containing the key input PAR of
%           simulation scenarios
%

function PAR = ak_fixed_wireless_set_parameters_outdoor_channels(scenario_idx)


if nargin < 1
    
    scenario_idx = 0;
    
end % if(nargin)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   set values for independent PAR of given  simulation scenarios       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch scenario_idx
    
    case 0 % default PAR
        
        
        %== Monte Carlo simulation PAR
        
        %   number of user location realizations
        PAR.num_ms_loc_realization = 1000;
        
        %   number of channel realizations per user location
        PAR.num_channel_realization = 100;
        
        %== network layout
        
        %   type of cells
        %   -   0: hexagonal cells
        %   -   1: circle cells
        PAR.cell_type = 0;
        
        %   inter-site distance in meters
        %   -   500m for urban micro
        %   -   1732m for suburban macro
        PAR.ISD = 200*sqrt(2);
        
        %   radius of cells in meters
        switch PAR.cell_type

            %   hexagonal cells
            case 0

                PAR.cell_radius = PAR.ISD / sqrt(3);

            %   circle cells
            case 1

                PAR.cell_radius = PAR.ISD / 2;

            otherwise

                errorMsg = 'This type of cells is not supported yet!';
                error(errorMsg);

        end % switch(PAR.cell_type)
        

        %   number of cells
        %   -   1: single cell without other-cell interference
        %   -   7, 19, 27
        % has to create patterns such as 121, etc:
        %network_sectorization_pattern =...
        %    PAR.num_cell * 1000 + PAR.num_sector_per_cell * 100 ...
        %    + PAR.coord_pattern * 10 + PAR.bs_cluster_per_sector;

        PAR.num_cell = 7;
        
        %   number of sectors per cell
        PAR.num_sector_per_cell = 1;
        
        %   number of users per sector
        PAR.num_ms_per_sector = 3;
        
        %   radius of the holes in the cells where users are not allowed to stay
        PAR.hole_radius = 35;
        
        %   sector type: 
        %       + 0: one-sector-per-cell hexagonal
        %       + 1: one-sector-per-cell circle
        %       + 2: three-sector-per-cell diamond
        PAR.sector_type = 0;
        
        
        %== coordination
        
        PAR.coord_pattern = 2;  %use 1 if num_cell = 19
        
        %   number of sectors per coordinated cluster 
        PAR.num_sector_per_coord_cluster = 1;
        
        
        
        %== FDD uplink
        
        %   FDD UL center frequency in Hz
        PAR.FDD_UL_freq = 3.5e9;
        
        PAR.FDD_UL_freq_GHz = PAR.FDD_UL_freq / 1e9;
        
        %   wavelength corresponding to center frequency
        PAR.FDD_UL_wavelength = 3e8 / PAR.FDD_UL_freq;
        
        %   wavenumber
        PAR.FDD_UL_wavenumber = 2 * pi / PAR.FDD_UL_wavelength;
        
        %   total bandwidth of a carrier in Hz
        %   -   typical values are 5e6, 10e6, 20e6
        PAR.FDD_UL_totalBWHz = 40e6;
        
        %   occupied bandwidth of a subcarrier in Hz
        PAR.FDD_UL_carrier_BWHz = PAR.FDD_UL_totalBWHz/1024; %180e3;
        
        
        %== FDD downlink
        
        %   FDD DL center frequency in Hz
        PAR.FDD_DL_freq = 3.8e9;
        
        PAR.FDD_DL_freq_GHz = PAR.FDD_DL_freq / 1e9;
        
        %   wavelength corresponding to center frequency
        PAR.FDD_DL_wavelength = 3e8 / PAR.FDD_DL_freq;
        
        %   wavenumber
        PAR.FDD_DL_wavenumber = 2 * pi / PAR.FDD_DL_wavelength;
        
        %   total bandwidth of a carrier in Hz
        %   -   typical values are 5e6, 10e6, 20e6
        PAR.FDD_DL_totalBWHz = 40e6;
        
        %   occupied bandwidth of a subcarrier in Hz
        PAR.FDD_DL_carrier_BWHz = PAR.FDD_DL_totalBWHz/1024; %180e3;
        
        
        %== TDD
        
        %   TDD center frequency in Hz
        PAR.TDD_freq = 1.8e9; %1.8e9 3.5 etc;
        
        PAR.TDD_freq_GHz = PAR.TDD_freq / 1e9;
        
        %   wavelength corresponding to center frequency
        PAR.TDD_wavelength = 3e8 / PAR.TDD_freq;
        
        %   wavenumber
        PAR.TDD_wavenumber = 2 * pi / PAR.TDD_wavelength;
        
        %   total bandwidth of a carrier in Hz
        %   -   typical values are 5e6, 10e6, 20e6
        PAR.TDD_totalBWHz = 2*40e6;
        
        %   occupied bandwidth of a subcarrier in Hz
        PAR.TDD_carrier_BWHz = PAR.TDD_totalBWHz/1024; %180e3;
        
        %== power
        
        %   thermal noise power density (PSD) in dBm/Hz, which is equal to
        %   3.98 x 10 ^(-19) mW/Hz
        PAR.thermal_noisePSD_dBmHz = -174;
        
        %   noise figure at each base station in dB/Hz
        PAR.bs_noise_figure_dBHz = 5;
        
        %   noise figure at each user in dB/Hz
        PAR.ms_noise_figure_dBHz = 9;
        
        
        %   transmit power at base stations in data transmission stage in dBm
        PAR.DL_data_tx_power_dBm = 46;
        
        %   transmit power at users in training stage in dBm
        PAR.UL_pilot_tx_power_dBm = 24;
        
        %   transmit power at users in data transmission stage in dBm
        PAR.UL_data_tx_power_dBm = 24;
        
        
        
        %== base stations
        
        %   number of base station antenna clusters in a sector
        PAR.bs_cluster_per_sector = 1;
        
        %   number of antennas per cluster at a sector
        PAR.bs_antenna_per_sector = 64;
        
        %   is there a base station antenna cluster at the cell center
        PAR.is_there_center_cluster = 1;
        
        %   type of base station antenna cluster distribution
        %   -   0: equally distributed on a ring
        %   -   1: uniformly distributed on a ring
        %   -   2: uniformly distributed in the cell
        PAR.bs_cluster_loc_type = 0;
       
        
        
        %== users
        
        %   number of antennas at a user
        %   -   KTT: only single-antenna users are currently supported
        PAR.ms_antenna = 1;
        
        %   type of user location distribution
        %   -   0: equally distributed on a ring
        %   -   1: uniformly distributed on a ring
        %   -   2: uniformly distributed in the cell
        PAR.ms_loc_type = 2;
        
        %   radius of the ring on which user(s) of interest are located
        %PAR.ms_ring_radius = 600;
        PAR.ms_ring_radius = NaN;
        
        %   user speed [m/s] - should be less than 50m/s corresponding to
        %   180km/h
        %   15km/h => 4.17m/s
        %   30km/h => 8.33m/s
        %   60km/h => 16.66mm/s
        %   120km/h => 33.33m/s
        %   180km/h => 50m/s
        PAR.ms_velocity = 8 * 1000 / 3600;
        
        
        %== channel
        
        %   channel model index
        PAR.channel_model_idx = 0;
        
        %   large scale fading model index
        %   -   0: PLNLOS(d) = 128.1 + 36.7 log10(d), d in km for fc = 2GHz and
        %   d > 35m
        PAR.pathloss_model_idx = 0;
        
        %   penetration loss
        PAR.penetration_loss_dB = 20;
        
        
        %   training and analog feedback lengths
        PAR.FDD_UL_training = PAR.num_ms_per_sector;
        
        PAR.TDD_UL_training = PAR.num_ms_per_sector;
        
        PAR.FDD_UL_feedback = PAR.num_ms_per_sector * PAR.bs_antenna_per_sector;
        
        PAR.FDD_DL_training = PAR.bs_antenna_per_sector;
        

        %== frame structure

        %   symbol duration
        %   LTE symbol duration is Tsymbol = 1/(15 kHz)
        PAR.symbol_duration = 1 / 300000;

    otherwise
        
        errorMsg = 'ERROR: This simulation scenario is not supported yet!';
        error(errorMsg);
        
end % switch(scenario_idx)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   set values for dependent PAR                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%== network layout

%   number of sectors
PAR.num_sector = PAR.num_cell * PAR.num_sector_per_cell;

%   number of base stations, each corresponding to one sector
PAR.num_bs = PAR.num_sector;

%   number of users in the whole network
PAR.num_ms = PAR.num_sector * PAR.num_ms_per_sector;

%   number of coordinated clusters
PAR.num_coord_cluster = PAR.num_sector / PAR.num_sector_per_coord_cluster;

%   number of users per coordinated clusters
PAR.num_ms_per_coord_cluster = PAR.num_ms / PAR.num_coord_cluster;

%   number of base station antenna clusters per coordinated cluster
PAR.bs_cluster_per_coord_cluster =...
    PAR.bs_cluster_per_sector * PAR.num_sector_per_coord_cluster;

%   number of base station antenna clusters per cell
PAR.bs_cluster_per_cell = PAR.bs_cluster_per_sector * PAR.num_sector_per_cell;

%   number of base station antenna clusters in the network
PAR.num_bs_cluster = PAR.num_coord_cluster * PAR.bs_cluster_per_coord_cluster;



%== base stations

%   number of base station antenna cluster per coordinated sector cluster
PAR.num_bs_cluster_per_coord_cluster =...
    PAR.bs_cluster_per_sector * PAR.num_sector_per_coord_cluster;

%   number of antennas per base station antenna cluster
PAR.bs_antenna_per_bs_cluster =...
    PAR.bs_antenna_per_sector / PAR.bs_cluster_per_sector;

%   number of base station antennas per coordinated sector cluster
PAR.bs_antenna_per_coord_cluster =...
    PAR.bs_antenna_per_bs_cluster * PAR.num_bs_cluster_per_coord_cluster;


%== users

%   indices of the users in the matrix form to make it easy to find the
%   users usign the same pilot. The matrix has size
%   -   PAR.num_ms_per_coord_cluster
%   -   PAR.num_coord_cluster
%   -   each row has the indices of the users sharing the same pilot with
%       each other
PAR.same_pilot_ms_indices =...
    reshape(1:PAR.num_ms, PAR.num_ms_per_coord_cluster, PAR.num_coord_cluster);

%   user velocity in km/hour
PAR.ms_velocity_kmh = PAR.ms_velocity * (3600 / 1000);



%== power

%   transmit power at base stations in data transmission stage in dB  
PAR.DL_data_tx_power_dB = PAR.DL_data_tx_power_dBm - 30;

%   transmit power at base stations in data trans. stage in linear scale
PAR.DL_data_tx_power = 10 .^ (PAR.DL_data_tx_power_dB / 10);


%   transmit power at users in data transmission stage in dB        
PAR.UL_data_tx_power_dB = PAR.UL_data_tx_power_dBm - 30;

%   transmit power at users in data transmission stage in linear scale
PAR.UL_data_tx_power = 10 .^ (PAR.UL_data_tx_power_dB / 10);

%   transmit power at users in training stage in dB        
PAR.UL_pilot_tx_power_dB = PAR.UL_pilot_tx_power_dBm - 30;

%   transmit power at users in training stage in linear scale
PAR.UL_pilot_tx_power = 10 .^ (PAR.UL_pilot_tx_power_dB / 10);



%   total noise density at each base station in dBm/Hz
PAR.bs_total_noisePSD_dBmHz =...
    PAR.thermal_noisePSD_dBmHz + PAR.bs_noise_figure_dBHz;

%   noise power at a base station per subcarrier in linear scale
PAR.FDD_UL_noise_variance =...
    (10 ^ ((PAR.bs_total_noisePSD_dBmHz - 30) / 10)) * PAR.FDD_UL_totalBWHz;

PAR.TDD_UL_noise_variance =...
    (10 ^ ((PAR.bs_total_noisePSD_dBmHz - 30) / 10)) * PAR.TDD_totalBWHz;

%   total noise figure at a mobile station in dBm/Hz
PAR.ms_total_noisePSD_dBmHz =...
    PAR.thermal_noisePSD_dBmHz + PAR.ms_noise_figure_dBHz;

%   noise power at a mobile station per subcarrier in linear scale
PAR.FDD_DL_noise_variance =...
    (10 ^ ((PAR.ms_total_noisePSD_dBmHz - 30) / 10)) * PAR.FDD_DL_totalBWHz;

PAR.TDD_DL_noise_variance =...
    (10 ^ ((PAR.ms_total_noisePSD_dBmHz - 30) / 10)) * PAR.TDD_totalBWHz;

%   ratio of noise variance at a base station over uplink data transmit power
PAR.FDD_UL_noise_to_signal_ratio =...
    PAR.FDD_UL_noise_variance / PAR.UL_data_tx_power;

PAR.TDD_UL_noise_to_signal_ratio =...
    PAR.TDD_UL_noise_variance / PAR.UL_data_tx_power;

%   ratio of noise variance at a user over downlink data transmit power
PAR.FDD_DL_noise_to_signal_ratio =...
    PAR.FDD_DL_noise_variance / PAR.DL_data_tx_power;

PAR.TDD_DL_noise_to_signal_ratio =...
    PAR.TDD_DL_noise_variance / PAR.DL_data_tx_power;


%== frame structure

%   symbol sampling rate
PAR.channel_sampling_rate = 1 / PAR.symbol_duration;

%== time-correlated fading channel generation

%   Doppler shift
PAR.FDD_UL_Doppler_shift = PAR.ms_velocity / PAR.FDD_UL_wavelength;

PAR.FDD_DL_Doppler_shift = PAR.ms_velocity / PAR.FDD_DL_wavelength;

PAR.TDD_Doppler_shift = PAR.ms_velocity / PAR.TDD_wavelength;

%   normalized Doppler frequency
PAR.FDD_UL_normalized_Doppler_shift =...
    PAR.FDD_UL_Doppler_shift * PAR.symbol_duration;

PAR.FDD_DL_normalized_Doppler_shift =...
    PAR.FDD_DL_Doppler_shift * PAR.symbol_duration;

PAR.TDD_normalized_Doppler_shift =...
    PAR.TDD_Doppler_shift * PAR.symbol_duration;

%   time frame is 1/(2 * f_D), where f_D is the normalized Doppler shift
PAR.FDD_UL_time_frame = 1 / (2 * PAR.FDD_UL_normalized_Doppler_shift);

PAR.FDD_DL_time_frame = 1 / (2 * PAR.FDD_DL_normalized_Doppler_shift);

PAR.TDD_time_frame = 1 / (2 * PAR.TDD_normalized_Doppler_shift);

PAR.time_frame_length =...
    min([PAR.FDD_UL_time_frame PAR.FDD_DL_time_frame PAR.TDD_time_frame]);

min_time_frame_length =...
    (PAR.num_ms_per_sector + 1) * (PAR.bs_antenna_per_coord_cluster + 1);

%   uplink actual data transmission in FDD massive MIMO
PAR.FDD_UL_data_Tx_time =...
    PAR.time_frame_length - PAR.FDD_UL_training - PAR.FDD_UL_feedback;

%   downlink actual data transmission in FDD massive MIMO
PAR.FDD_DL_data_Tx_time =...
    PAR.time_frame_length - PAR.FDD_DL_training - PAR.FDD_UL_feedback;

PAR.FDD_noCSIwaiting_DL_data_Tx_time =...
    PAR.time_frame_length - PAR.FDD_DL_training;

%   uplink actual data transmission in TDD massive MIMO
PAR.TDD_UL_data_Tx_time =...
    floor((PAR.time_frame_length - PAR.TDD_UL_training) / 2);

%   downlink actual data transmission in TDD massive MIMO
PAR.TDD_DL_data_Tx_time = PAR.TDD_UL_data_Tx_time;

% if PAR.time_frame_length < min_time_frame_length
%     
%     errMsg = 'ERROR: The time frame length is too small!';
%     error(errMsg);
%     
% end

end % function 