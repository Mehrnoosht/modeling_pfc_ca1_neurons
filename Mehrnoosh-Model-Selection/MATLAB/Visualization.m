%% This script is for History dependent model 

clear;close all;clc 

%% Load data 

lin_pos = load('Data/3-2-4-3/linear_position.mat');
spike = load('Data/3-2-4-3/spike.mat');
directionn = load('Data/3-2-4-3/direction.mat');
speed = load('Data/3-2-4-3/speed.mat');
time = load('Data/3-2-4-3/time.mat');
% phi = load('Data/3-2-4-3/phi.mat');

time = time.struct.time;
lin_pos = lin_pos.struct.linear_distance';
speed = speed.struct.speed';
directionn = directionn.struct.head_direction';
spike = spike.struct.is_spike';
% phi = phi.phi';

lin_pos(isnan(lin_pos))=0;speed(isnan(lin_pos))=0;
directionn(isnan(directionn))=0;spike = double(spike);

thsh = 15;
%% Emperical Model

N = length(spike);
bins = min(lin_pos):10:max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = count./occupancy;
%% Lin_pos-Speed
figure;
scatter(lin_pos,speed);hold on
scatter(lin_pos(spike==1),speed(spike==1),'.r');
xlabel('lin-pos[cm]');ylabel('speed[cm/s]');
saveas(gcf,[pwd '/Results/R-3-2-4-3/speed-linpos.fig']);
saveas(gcf,[pwd '/Results/R-3-2-4-3/speed-linpos.png']);
%% Theta_linpos with direction

direction = diff(lin_pos);
direction(direction>=0)=1; % Moving out
direction(direction<0)=0; % Moving in

figure;
subplot(2,2,1)
scatter(lin_pos(spike(2:end)==1 & direction==0),phi(spike(2:end)==1 & direction==0));
hold on
scatter(lin_pos(spike(2:end)==1 & direction==1),phi(spike(2:end)==1 & direction==1),'r');
xlabel('lin-pos[cm]');ylabel('Theta Phase');legend('Inbound','Outbound')

subplot(2,2,2)
scatter(lin_pos(spike(2:end)==1 & direction==0),amp(spike(2:end)==1 & direction==0));
hold on
scatter(lin_pos(spike(2:end)==1 & direction==1),amp(spike(2:end)==1 & direction==1),'r');
xlabel('lin-pos[cm]');ylabel('Theta Amplitude');legend('Inbound','Outbound')

subplot(2,2,3)
scatter(lin_pos(spike(2:end)==1 & speed(2:end)>=thsh & direction==0),...
    phi(spike(2:end)==1 & speed(2:end)>=thsh & direction==0));
hold on 
scatter(lin_pos(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),...
    phi(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),'r');
xlabel('lin-pos[cm]');ylabel('Theta Phase');legend('Inbound','Outbound')
title('Theta Precession for V>thsh[cm/s]')

subplot(2,2,4)
scatter(lin_pos(spike(2:end)==1 & speed(2:end)>=thsh & direction==0),...
    amp(spike(2:end)==1 & speed(2:end)>=thsh & direction==0));
hold on 
scatter(lin_pos(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),...
    amp(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),'r');
xlabel('lin-pos[cm]');ylabel('Theta Amplitude');legend('Inbound','Outbound')
title('Theta Precession for V>thsh[cm/s]')

saveas(gcf,[pwd '/Results/R-3-2-4-3/Theta_linpos_dir.fig']);
% saveas(gcf,[pwd '/Results/R-3-2-4-3/Theta_linpos_dir.png']);
%% Theta_speed with direction
figure;
subplot(2,2,1)
scatter(speed(spike(2:end)==1 & direction==0),phi(spike(2:end)==1 & direction==0));
hold on
scatter(speed(spike(2:end)==1 & direction==1),phi(spike(2:end)==1 & direction==1),'r');
xlabel('speed[cm/s]');ylabel('Theta Phase');legend('Inbound','Outbound')

subplot(2,2,2)
scatter(speed(spike(2:end)==1 & direction==0),amp(spike(2:end)==1 & direction==0));
hold on
scatter(speed(spike(2:end)==1 & direction==1),amp(spike(2:end)==1 & direction==1),'r');
xlabel('speed[cm/s]');ylabel('Theta Amplitude');legend('Inbound','Outbound')

subplot(2,2,3)
scatter(speed(spike(2:end)==1 & speed(2:end)>=thsh & direction==0),...
    phi(spike(2:end)==1 & speed(2:end)>=thsh & direction==0));
hold on 
scatter(speed(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),...
    phi(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),'r');
xlabel('speed[cm/s]');ylabel('Theta Phase');legend('Inbound','Outbound')
title('Theta Precession for V>thsh[cm/s]')

subplot(2,2,4)
scatter(speed(spike(2:end)==1 & speed(2:end)>=thsh & direction==0),...
    amp(spike(2:end)==1 & speed(2:end)>=thsh & direction==0));
hold on 
scatter(speed(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),...
    amp(spike(2:end)==1 & speed(2:end)>=thsh & direction==1),'r');
xlabel('speed[cm/s]');ylabel('Theta Amplitude');legend('Inbound','Outbound')
title('Theta Precession for V>thsh[cm/s]')

saveas(gcf,[pwd '/Results/R-3-2-4-3/Theta_speed_dir.fig']);
% saveas(gcf,[pwd '/Results/R-3-2-4-3/Theta_speed_dir.png']);