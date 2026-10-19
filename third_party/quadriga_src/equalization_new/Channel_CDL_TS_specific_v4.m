function [h_f,PriorIdeal] = Channel_CDL_TS_specific_v4(type, N_sc, center_frequency, mobile_speed, RMS_delay, N_rx, N_ry, K , duration,NClusters)
% -------------------------------------- To test score-based method -------------------------------------- %
% Generate the frequency channel response of the users in the cell and the
% knonwn interference and the active unknown interference

%  lambdac = physconst('LightSpeed')/center_frequency; % 波长 
%  SD = lambdac*14e3/mobile_speed; % 半波长采样点数
% rng('default') ;

% Input :
% type : channel model following TS 38.902
% N_sc : number of the OFDM subcarriers
% center_frequency : Center frequency of the electromagnetic wave
% mobile_speed : speed of the users [m/s]
% RMS_delay : r.m.s. path delay spread [ns]
% N_r : number of the Rx antenna number
% K_active : number of the active users (including interference) 
% ------ example ------
%  clear ;
%  clc ;
%  type = 'NR-CDL-B' ;
%  N_sc = 24 * 12 ;
%  center_frequency = 6e9 ; % Set center frequency to 6 GHz ---- must change f in generate together
%  mobile_speed = 0 ;  % Mobile sppeed : 0 m/s
%  RMS_delay = 100 ; 
%  K_active = 1 ;  % number of active MTs
%  N_r = 64 ;
 lambdac = physconst('LightSpeed')/center_frequency; % 波长 
 % MT parameters 实际上为发端，但发端无法设置number > 1，所以用no_rx函数
 L = qd_layout ;
 N_t = 1 ;
 L.no_rx = K / N_t;  % Set number of MTs
 L.randomize_rx_positions( 300 , 1.5, 1.5, 0 , [] ,100 ) ; % Random position with cell radius = 200 m
%  for k = 1 : 1
%     theta = rand(1) * 120 + 30 ;
%     distance = 20 + rand(1) * 80 ;
%     L.rx_position(:,k) = [distance*sind(theta) ; -distance*cosd(theta) ;  1.5] ;
%  end
 % for k = K/2+1 : K
 %    theta = rand(1) * 120 + 120 ;
 %    distance = 300 + rand(1) * 700 ;
 %    L.rx_position(:,k) = [distance*cosd(theta) ; distance*sind(theta) ; 1.5] ;
 % end
 L.rx_array = qd_arrayant('3gpp-3d',1, 1, center_frequency, 1) ;
i_type = 1 ;
b = qd_builder.gen_cdl_model( type(i_type,:), center_frequency, mobile_speed(1), duration, RMS_delay(1) , [], [], [], [], [], []) ; 
if strcmp(type(i_type,:),'NR-CDL-B') == 1
    NumClusters = 24 ; % NR-CDL-B
elseif strcmp(type(i_type,:),'NR-CDL-C') == 1
    NumClusters = 25; % NR-CDL-C
    % NumClusters = NClusters ;
elseif strcmp(type(i_type,:),'NR-CDL-A') == 1
    NumClusters = 24 ; % NR-CDL-A  
end
NumSubpath_PerCluster = 20;
% BS parameters
rx_trk = qd_track ;
rx_trk.initial_position = [0 ; 0 ; 25] ; % BS : 25 m height 
b.rx_track = rx_trk ;
b.rx_positions = rx_trk.initial_position ;
b.rx_array = qd_arrayant('3gpp-3d',N_rx, N_ry, center_frequency, 1) ; % BS antennas


% Obtain the channel coefficients by Quadriag toolbox
for k = 1 : K / N_t
    %  duration = 0 ;   % Channel observation time : 1 mu s
    % SD = lambdac*14e3/mobile_speed(k); % 半波长采样点数
    if strcmp(type(i_type,:),'NR-CDL-A') == 1
        aod = [ 0, -178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83, 134.8, -153,...
            -172, -129.9, -136, 165.4, 148.4, 132.7, -118.6, -154.1, 126.5, -56.2 ] + ( rand(1, 24) * 360 - 180) ;
        aoa = [ -180, 51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.8, -41.9, 94.2, 51.9, -115.9, ...
            26.6, 76.6, -7, -23, -47.2, 110.4, 144.5, 155.3, 102, -151.8, 55.2 ]  + ( rand(1, 24) * 360 - 180) ;
        zod = [ 98.5, 50.2, 93.2, 93.2, 93.2, 122, 122, 122, 150.2, 55.2, 26.4, 126.4, 171.6, 151.4, 157.2,...
            47.2, 40.4, 43.3, 161.8, 10.8, 16.7, 171.7, 22.7, 144.9 ] + ( rand(1, 24) * 30 - 15);
        zoa = [ 81.5, 125.4, 91.3, 91.3, 91.3, 94, 94, 94, 47.1, 56, 30.1, 58.8, 26, 49.2, 143.1, 117.4,...
            122.7, 123.2, 32.6, 27.2, 15.2, 146, 150.7, 156.1 ] + ( rand(1, 24) * 30 - 15);
    elseif strcmp(type(i_type,:),'NR-CDL-B') == 1
        aod = [ 0, 9.3, 9.3, 9.3, -34.1, -65.4, -11.4, -11.4, -11.4, -67.2, 52.5, -72, 74.3, -52.2, ...
            -50.5, 61.4, 30.6, -72.5, -90.6, -77.6, -82.6, -103.6, 75.6, -77.6 ]  + ( rand(1, 24) * 360 - 180) ;
        aoa = [ -180, -173.3, -173.3, -173.3, 125.5, -88.0, 155.1, 155.1, 155.1, -89.8, 132.1, ...
            -83.6, 95.3, 103.7, -87.8, -92.5, -139.1, -90.6, 58.6, -79.0, 65.8, 52.7, 88.7, -60.4 ]  + ( rand(1, 24) * 360 - 180) ;
        zod = [ 98.5, 105.8, 105.8, 105.8, 115.3, 119.3, 103.2, 103.2, 103.2, 118.2, 102.0, 100.4, ...
             98.3, 103.4, 102.5, 101.4, 103.0, 100.0, 115.2, 100.5, 119.6, 118.7, 117.8, 115.7 ]  + ( rand(1, 24) * 30 - 15);
        zoa = [ 81.5, 78.9, 78.9, 78.9, 63.3, 59.9, 67.5, 67.5, 67.5, 82.6, 66.3, 61.6, 58.0, 78.2, ...
             82.0, 62.4, 78.0, 60.9, 82.9, 60.8, 57.3, 59.9, 60.1, 62.3 ] + ( rand(1, 24) * 30 - 15);    
    elseif strcmp(type(i_type,:),'NR-CDL-C') == 1
        aod = [ 0, -46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3, 73.1, -64.5, 80.2, -97.1, -55.3, ...
            -64.3, -78.5, 102.7, 99.2, 88.8, -101.9, 92.2, 93.3, 106.6, 119.5, -123.8 ] + ( rand(1, 25) * 360 - 180) ;
        aoa = [ -180, -101, 120, 120, 120, -127.5, 170.4, 170.4, 170.4, 55.4, 66.5, -48.1, 46.9, ...
            68.1, -68.7, 81.5, 30.7, -16.4, 3.8, -13.7, 9.7, 5.6, 0.7, -21.9, 33.6 ] + ( rand(1, 25) * 360 - 180) ;
        zod = [ 98.5, 97.2, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 105.2, 95.3, 106.1, 93.5, 103.7, ...
            104.2, 93.0, 104.2, 94.9, 93.1, 92.2, 106.7, 93.0, 92.9, 105.2, 107.8 ] + ( rand(1, 25) * 30 - 15);
        zoa = [ 81.5, 87.6, 72.1, 72.1, 72.1, 70.1, 75.3, 75.3, 75.3, 67.4, 63.8, 71.4, 60.5, 90.6, ...
            60.1, 61.0, 100.7, 62.3, 66.7, 52.9, 61.8, 51.9, 61.7, 58, 57 ] + ( rand(1, 25) * 30 - 15);
    else
        error('Unknown channel model')
    end
    ang = [ aod; aoa; 90-zod; 90-zoa ] * pi / 180;
    b.AoD = ang(1,:); b.AoA = ang(2,:); b.EoD = ang(3,:); b.EoA = ang(4,:);

    % Get random initial position of the MT
    b.tx_position = L.rx_position(:,k) ;
    b.tx_array = L.rx_array(k) ;
    b.tx_track = L.rx_track(k) ;
    b.tx_track.orientation(3) = b.tx_track.orientation(3) * -1 ;

    % * obtain channel coefficient

    c = b.get_channels ; % Generate channels
    N_r = N_rx*N_ry;
    h_physical(:,:,:,k) = reshape( c.coeff , N_r , N_t , NumClusters , 1 );
end

        pow_path = squeeze(mean(abs(h_physical).^2,[1,2])) ;
        Delay = reshape( c.delay , [1,1,NumClusters,1,1] ) ;
        clear L ;        
        sc_idx = reshape( 0:N_sc-1,[1,1,1,1,N_sc] );  
        h_k = h_physical .* exp( -1i * 2 * pi * 30 * 1e3 .* Delay .* sc_idx ) ; % Nr by Nt by Npath by K by Nsc
        h_f0 = permute( sum(h_k,3),[5 1 2 4 3] ) ; % Nsc * Nr * Nt * K
        pow = mean(abs(h_f0).^2,[1,2]) ;
        pow_obr = squeeze(pow) ;
        h_f0 = h_f0 ./ sqrt(pow) ; % normalization
        pow_cali = squeeze(mean(abs(h_f0).^2,[1,2])) ;
        for i_k = 1 : K/N_t
                h_f(:,:,1,i_k*2-1) = squeeze(h_f0(:,:,1,i_k) );
                % h_f(:,:,1,i_k*2) = squeeze(h_f0(:,:,2,i_k) );         
        end
        % h_f_obr = h_f.' ;
        % h_obr1 = ifft(h_f_obr);
        % figure(1)
        % stem(mean(abs(h_obr1).^2,2));
        % h_obr2 = fft(h_f_obr);
        % figure(2)
        % stem(mean(abs(h_obr2).^2,2));

%         % 频域导向矢量，空域导向矢量，及path gain  
        af_steer_cluster =  exp( -1j*2*pi*30*1e3.*(0:N_sc-1)' .* reshape( Delay(:,:,2:end),[1,NumClusters-1]) ) ;
        af_steer = kron(af_steer_cluster , ones(1,NumSubpath_PerCluster)) ; % N_sc by  N_path 
        as_steer = c.steer(:,2:end)  ; % N_r by  N_path 
        as_steer_reshape = reshape(as_steer,N_rx,N_ry,[]);
%         as_steer_reshape = reshape(as_steer,N_ry,N_rx,[]);
%         plot(unwrap(angle(as_steer_reshape(:,3,2))));
%         plot(unwrap(angle(as_steer_reshape(1,:,2))));
        
        angle_X = wrapTo2Pi(diff(angle(as_steer_reshape),1,1));
        angle_X = squeeze(mean(angle_X,[1,2]));
        angle_Y = wrapTo2Pi(diff(angle(as_steer_reshape),1,2));
        angle_Y = squeeze(mean(angle_Y,[1,2]));
%         % 重构导向矢量以确定正确
%         for pathi = 1:length(angle_X)
%             as_steer_reshape_rebuild(:,:,pathi) = as_steer_reshape(1,1,pathi)...
%                 .*exp(1j*[0:size(as_steer_reshape,1)-1].'*angle_X(pathi))...
%                 *exp(1j*[0:size(as_steer_reshape,2)-1]*angle_Y(pathi));
%         end
%         sum((abs(as_steer_reshape_rebuild - as_steer_reshape)).^2,'all');
        
%         reshape(b.rx_array.element_position(2,:),16,64);
%         reshape(b.rx_array.element_position(3,:),16,64);
%         for i = 1:16
%             plot(b.rx_array.element_position(2,i:16:end));
%         end
%         
%         for i = 1:64
%             plot(b.rx_array.element_position(3,(i-1)*16+1:i*16));
%         end
        rho = squeeze(mean(c.g_sub_fnl(:,1,2:end),1)) / pow(:,:,1) ; % 1 by N_path
        PriorIdeal.af_steer = af_steer ;
        PriorIdeal.as_steer = as_steer_reshape ;
        PriorIdeal.rho = rho;
        PriorIdeal.angle_X = angle_X;
        PriorIdeal.angle_Y = angle_Y;
%         PriorIdeal = 0;
end
