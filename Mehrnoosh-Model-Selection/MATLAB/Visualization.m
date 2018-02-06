%% This script is for History dependent model 

clear 
close all
clc 

%% Load data 

lin_pos = load('Data/Tetrode5,Neuron2/linear_position.mat');
spike = load('Data/Tetrode5,Neuron2/spike.mat');
direction = load('Data/Tetrode5,Neuron2/direction.mat');
speed = load('Data/Tetrode5,Neuron2/speed.mat');
time = load('Data/Tetrode5,Neuron2/time.mat');
phi = load('Data/Tetrode5,Neuron2/phi.mat');
amp = load('Data/Tetrode5,Neuron2/amp.mat');
%lfp = load('Data/Tetrode5,Neuron2/lfp_lo_th/.mat');

time = time.struct.time;
lin_pos = lin_pos.struct.linear_distance;
lin_pos = lin_pos';
speed = speed.struct.speed;
speed = speed';
direction = direction.struct.head_direction;
direction = direction';
spike = spike.struct.is_spike;
spike = spike';
amp = amp.amp;
amp = amp';
phi = phi.phi;
phi = phi';

lin_pos(isnan(lin_pos))=0;
speed(isnan(lin_pos))=0;
direction(isnan(direction))=0;
spike = double(spike);

%% Emperical Model

N = length(spike);
bins = min(lin_pos):10:max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = count./occupancy;
%% Visualization

% Lin_pos-Speed
figure;
scatter(lin_pos,speed);
hold on
scatter(lin_pos(spike==1),speed(spike==1),'.r');
xlabel('lin-pos[cm]');
ylabel('speed[cm/s]');

% Theta_Rythem-Speed
figure;
subplot(2,2,1) 
scatter(lin_pos(spike==1),phi(spike==1));
xlabel('lin-pos[cm]')
ylabel('Theta Phase')

subplot(2,2,2);
scatter(lin_pos(spike==1),amp(spike==1));
xlabel('lin-pos[cm]')
ylabel('Theta Amplitude')

subplot(2,2,3) 
scatter(speed(spike==1),phi(spike==1));
xlabel('speed[cm/s]')
ylabel('Theta Phase')

subplot(2,2,4);
scatter(speed(spike==1),amp(spike==1));
xlabel('speed[cm/s]')
ylabel('Theta Amplitude')




