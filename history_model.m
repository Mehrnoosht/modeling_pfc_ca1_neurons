%% This script is for History dependent model 

clear 
close all
clc 

%% Load data 

data_pos = csvread('\Mehrnoosh-Model-Selection\MATLAB\Data\linear_position.csv');
data_spike = csvread('spike.csv');
time = data_pos(:,1);
lin_pos = data_pos(:,2);
spike = data_spike(:,2);
 
%% Autocorrelation 

acf = xcorr(spike-mean(spike),'coeff');
figure;
stem(1:1000,acf(1:1000))
ylabel('Autocorrelation')
xlabel('Time')

%% Emperical Model
N = length(spike);
bins = min(lin_pos):10:max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = count./occupancy;

%% History Component

T = length(time);
lag = 150;
Hist = [];
for i=1:lag
   Hist = [Hist spike(lag-i+1:end-i)];
end

y=spike(lag+1:end);
%% Indicator basis function 

[b,dev,stats] = glmfit(Hist,y,'poisson');

one = ones(length(Hist),1);
X = [one Hist];
lambda = exp(X*b);

%% GLM (Spline)

c_pt = [-10 0 20 40 60 80 100 120 140 150 160];
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
%% Gaussian RBf

mu = linspace(0,100,10);
design_matrix_rbf = [];
n = length(mu);

for i=1:n
   mean = mu(i)*ones(lag,1); 
   design_matrix_rbf(:,i) = gaussian_rbf(spike(1:lag),mean,140) ; 
end

X_rbf = Hist*design_matrix_rbf;
[b_rbf,dev_rbf,stats_rbf] = glmfit(X_rbf,y,'poisson');

one = ones(length(y),1);
X = [one X_rbf];
lambda_rbf = exp(X*b_rbf);

%% KS plot

% Indicator
figure;
subplot(3,1,1)
KS =  KSplot(lambda,y);
title('KS Plot for Indicator function')
xlabel('Model CDF')
ylabel('Emperical CDF')

% Gaussian RBF
%figure;
subplot(2,1,1)
KS =  KSplot(lambda_rbf,y);
title('KS Plot for Gaussian RBF')
xlabel('Model CDF')
ylabel('Emperical CDF')

% Cardinal Spline 
figure;
%subplot(2,1,2)
KS_spl =  KSplot(lambda_spl,y);
title('KS Plot for Cardinal Spline')
xlabel('Model CDF')
ylabel('Emperical CDF')

%% Cumulative Residual Analysis

R_ind = cumsum(stats.resid); 
R_spl = cumsum(stats_spl.resid);
R_rbf = cumsum(stats_rbf.resid);
figure;
plot(time(lag+1:end),R_ind)
hold on
plot(time(lag+1:end),R_spl)
hold on
plot(time(lag+1:end),R_rbf)
title('Residual for History-dependent model')
xlabel('Time')
ylabel('Residual')
legend('Indicator','Cardinal Spline','Gaussian RBF')
grid


%% Models

% GLM (Spline)

c_pt_pos = [-210,-190,-150,-140,-120,-100,-90,-50,-20,0,20,40,60,80,120,150,200,220];
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

% desgn_hist_ind = [spline(lag+1:end,:) Hist];
% [b_hist_pos,dev_hist_pos,stat_hist_pos] = glmfit(desgn_hist_ind, y,'poisson');
% [yhat_pos,ylo_pos,yhi_pos] = glmval(b_hist_pos,desgn_hist_ind,'log',stat_hist_pos);
% lambda_hist_pos = yhat_pos;
%% KS plot

% Cardinal Spline 
figure;
subplot(2,1,1)
KS_spl =  KSplot(lambda_spl,y);
title('KS Plot for spline function on history component')
xlabel('Model CDF')
ylabel('Emperical CDF')

subplot(2,1,2)
KS_spl =  KSplot(lambda_spl_pos,y);
title('KS Plot for History dependent mdoel')
xlabel('Model CDF')
ylabel('Emperical CDF')


%% 
figure;

x = exp(b_spl);
plot(x)

figure;
plot(time,spike,time(lag+1:end),lambda_spl_pos);
