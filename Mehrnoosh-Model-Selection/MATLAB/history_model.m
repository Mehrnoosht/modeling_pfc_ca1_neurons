%% This script is for History dependent model 

clear 
close all
clc 

%% Load data 

lin_pos = load('Data/linear_position.mat');
spike = load('Data/spike.mat');
direction = load('Data/direction.mat');
time = load('Data/time.mat');

time = time.struct.time;
lin_pos = lin_pos.struct.linear_distance;
lin_pos = lin_pos';
direction = direction.struct.head_direction;
direction = direction';
spike = spike.struct.is_spike;

lin_pos(isnan(lin_pos))=0;
direction(isnan(direction))=0;
spike = double(spike');

%% Emperical Model

N = length(spike);
bins = min(lin_pos):10:max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = count./occupancy;

%% History Component

T = length(time);
lag = 200;
Hist = zeros(length(spike)-lag,lag);
for i=1:lag
   Hist(:,i) = spike(lag-i+1:end-i);
end

y=spike(lag+1:end);
%% Indicator basis function 

[b,dev,stats] = glmfit(Hist,y,'poisson');

one = ones(length(Hist),1);
X = [one Hist];
lambda = exp(X*b);

%% GLM (Spline) for History Component 

c_pt = [-10 0 20 40 60 80 100 120 140 160 180 200 210];
num_c_pts = length(c_pt);
s = 0.4; 
X_spl = zeros(lag,length(c_pt));
tic
for i=1:lag
    nearest_c_pt_index = max(find(c_pt<i));
    nearest_c_pt_time = c_pt(nearest_c_pt_index);
    next_c_pt_time = c_pt(nearest_c_pt_index+1);
    u = (i-nearest_c_pt_time)./(next_c_pt_time-nearest_c_pt_time);
    p=[u^3 u^2 u 1]*[-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];
    X_spl(i,nearest_c_pt_index-1:nearest_c_pt_index+2) = p; 
end
toc
X_hist_spl = Hist*X_spl;

[b_spl dev_spl stats_spl] = glmfit(X_hist_spl,y,'poisson');
one = ones(length(y),1);
X_spl_hist = [one X_hist_spl];
lambda_spl = exp(X_spl_hist*b_spl);

%% Visualization of spline basis functions for history

figure;
plot(1:lag,exp(X_spl*b_spl(2:end)))
xlabel('Lags')
title('Spline basis function for history')
grid
saveas(gcf,[pwd '/Results/Spline_module_history.fig']);
saveas(gcf,[pwd '/Results/Spline_module_history.png']);


%% GLM model based on lin_pos and history component

% GLM (Spline) for lin_pos
c_pt_pos = [-20,-10,0,10,15,20,30,50,60,70,80,90,100,110,130,140,150,170,190,200];
num_c_pts = length(c_pt_pos);
s = 0.4; 
spline = zeros(length(lin_pos),length(c_pt_pos));
tic
for i=1:length(lin_pos)
    nearest_c_pt_index = max(find(c_pt_pos<lin_pos(i)));
    nearest_c_pt_time = c_pt_pos(nearest_c_pt_index);
    next_c_pt_time = c_pt_pos(nearest_c_pt_index+1);
    u = (lin_pos(i)-nearest_c_pt_time)./(next_c_pt_time-nearest_c_pt_time);
    p=[u^3 u^2 u 1]*[-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];
    spline(i,:) = [zeros(1,nearest_c_pt_index-2) p zeros(1,num_c_pts-4-(nearest_c_pt_index-2))];
end
toc

design_matrix_hist = [spline(lag+1:end,:) X_hist_spl];
[b_spl_pos,dev_spl_pos,stat_spl_pos] = glmfit(design_matrix_hist, y,'poisson');
[yhat_pos,ylo_pos,yhi_pos] = glmval(b_spl_pos,design_matrix_hist,'log',stat_spl_pos);
lambda_spl_pos = yhat_pos;

%% Visualization for HistoryModel

figure;
plot(time,spike,time(lag+1:end),lambda_spl_pos);
title('Spline basis function for HistoryModel')
xlabel('Lag')
grid
saveas(gcf,[pwd '/Results/Spline_module_HistoryModel.fig']);
saveas(gcf,[pwd '/Results/Spline_module_HistoryModel.png']);

%% KS plot

% Cardinal Spline 
figure;
subplot(2,1,1)
KS_spl =  KSplot(lambda_spl,y);
title('KS Plot for spline function on history component')
xlabel('Model CDF')
ylabel('Emperical CDF')

subplot(2,1,2)
%figure;
KS_spl =  KSplot(lambda_spl_pos,y);
title('KS Plot for History dependent mdoel')
xlabel('Model CDF')
ylabel('Emperical CDF')

saveas(gcf,[pwd '/Results/KS_Plot_HistoryModel.fig']);
saveas(gcf,[pwd '/Results/KS_Plot_HistoryModel.png']);

%% Direction

direction(direction>=0)=1;
direction(direction<0)=0;

design_matrix_pdh = [design_matrix_hist direction(lag+1:end,:)];
[b_spl_dir,dev_spl_dir,stat_spl_dir] = glmfit(design_matrix_pdh, y,'poisson');
[yhat_dir,ylo_dir,yhi_dir] = glmval(b_spl_dir,design_matrix_pdh,'log',stat_spl_dir);
lambda_spl_dir = yhat_dir;

figure;
subplot(2,1,1)
KS_spl =  KSplot(lambda_spl_pos,y);
title('KS Plot for History dependent mdoel')
xlabel('Model CDF')
ylabel('Emperical CDF')

subplot(2,1,2)
KS_spl =  KSplot(lambda_spl_pos,y);
title('KS Plot for History dependent mdoel with direction')
xlabel('Model CDF')
ylabel('Emperical CDF')

%% 
 1-chi2cdf(dev_spl-dev_spl_pos,num_c_pts);
 