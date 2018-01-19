% This script is for fitting different model to spiking activity of a
% single neuron in rat hippocamus, CA1

clear 
close all
clc

%% Load data 

data_pos = csvread('linear_position.csv');
data_spike = csvread('spike.csv');
time = data_pos(:,1);
lin_pos = data_pos(:,2);
spike = data_spike(:,2);

%% Emperical Model
N = length(spike);
bins = min(lin_pos):10:max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = count./occupancy;
%% Linear Regression

design_matrix_reg = [lin_pos'];
[b_reg,dev_reg,stats_reg] = glmfit(design_matrix_reg,spike,'poisson');
lambda_reg = exp(b_reg(1)+b_reg(2)*design_matrix_reg);

%% GLM (Quadratic)

design_matrix_quad = [];
design_matrix_quad(:,1) = lin_pos';
design_matrix_quad(:,2) =(lin_pos.^2)';
[b_quad,dev_quad,stats_quad] = glmfit(design_matrix_quad,spike,'poisson');
lambda_quad = exp(b_quad(1)+b_quad(2)*design_matrix_quad(:,1)+b_quad(3)*design_matrix_quad(:,2));

%% GLM (RBF)

mu = linspace(min(lin_pos),max(lin_pos),17);
design_matrix_rbf = [];
n = length(mu);

for i=1:n
   mean = mu(i)*ones(length(lin_pos),1); 
   design_matrix_rbf(:,i) = gaussian_rbf(lin_pos,mean,140) ;
    
end

[b_rbf,dev_rbf,stats_rbf] = glmfit(design_matrix_rbf,spike,'poisson');

one = ones(length(lin_pos),1);
X = [one design_matrix_rbf];
lambda_rbf = exp(X*b_rbf);

%% GLM (Spline)

c_pt = [-210,-190,-150,-140,-120,-100,-90,-50,-20,0,20,40,60,80,120,150,200,220];
num_c_pts = length(c_pt);
s = 0.4; 
X = zeros(length(lin_pos),length(c_pt));
tic
for i=1:length(lin_pos)
    nearest_c_pt_index = max(find(c_pt<lin_pos(i)));
    nearest_c_pt_time = c_pt(nearest_c_pt_index);
    next_c_pt_time = c_pt(nearest_c_pt_index+1);
    u = (lin_pos(i)-nearest_c_pt_time)./(next_c_pt_time-nearest_c_pt_time);
    p=[u^3 u^2 u 1]*[-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];
    X(i,:) = [zeros(1,nearest_c_pt_index-2) p zeros(1,num_c_pts-4-(nearest_c_pt_index-2))];
end
toc
X_spl = X;

[b_spl,dev_spl,stat_spl] = glmfit(X_spl, spike,'poisson','constant','off');
[yhat,ylo,yhi] = glmval(b_spl,X_spl,'log',stat_spl,'constant','off');
lambda_spl = yhat;
%% Visualization

figure;
bar(bins,rate);
hold on
plot(lin_pos,lambda_reg,'Linewidth',2)
hold on 
plot(lin_pos,lambda_quad,'Linewidth',2)
hold on 
plot(lin_pos,lambda_rbf,'Linewidth',2)
hold on
plot(lin_pos,lambda_spl,'Linewidth',2)
xlabel('linear Position[cm]')
ylabel('Firing Rate')
legend('occupancy normalized','linear regression','Quadratic Poly.','Gaussian-RBF','Spline')

%% OR 
subplot(2,2,1)
bar(bins,rate);
hold on
plot(lin_pos,lambda_reg,'Linewidth',2)
xlabel('linear Position[cm]')
ylabel('Firing Rate[spikes/ms]')
title('Linear Regression GLM')

subplot(2,2,2)
plot(lin_pos,lambda_quad,'Linewidth',2)
hold on 
bar(bins,rate);
xlabel('linear Position[cm]')
ylabel('Firing Rate[spike/ms]')
title('Quadratic Function GLM')

subplot(2,2,3)
plot(lin_pos,lambda_rbf,'r','Linewidth',2)
hold on
bar(bins,rate);
xlabel('linear Position[cm]')
ylabel('Firing Rate[spike/ms]')
title('Gaussian RBF GLM')

subplot(2,2,4)
plot(lin_pos,lambda_spl,'r','Linewidth',2)
hold on 
bar(bins,rate);
xlabel('linear Position[cm]')
ylabel('Firing Rate[spike/ms]')
title('Cardinal Spline GLM')

%% KS plot 

% Linear Regreesion 
figure;
subplot(2,2,1)
KS_reg = KSplot(lambda_reg,spike);
title('KS Plot for Linear Regression')
xlabel('Model CDF')
ylabel('Emperical CDF')

% Quadratic Poly. 
subplot(2,2,2)
KS_quad = KSplot(lambda_quad,spike);
title('KS Plot for Quadratic Polynomial')
xlabel('Model CDF')
ylabel('Emperical CDF')

% RBF Gaussian
subplot(2,2,3)
KS_rbf =  KSplot(lambda_rbf,spike);
title('KS Plot for Gaussian RBF')
xlabel('Model CDF')
ylabel('Emperical CDF')

% Cardinal Spline 
lambda_spl = yhat;
subplot(2,2,4)
KS_spl =  KSplot(lambda_spl,spike);
title('KS Plot for Cardinal Spline')
xlabel('Model CDF')
ylabel('Emperical CDF')

%% AIC model

% Linear Regression
ll_reg = sum(log(poisspdf(spike,lambda_reg')));
AIC_reg = -2*ll_reg+2*2;

% Quadratic Poly. 
ll_quad = sum(log(poisspdf(spike,lambda_quad)));
AIC_quad = -2*ll_quad+2*2;

% Gaussian RBF
ll_rbf = sum(log(poisspdf(spike,lambda_rbf)));
AIC_rbf = -2*ll_rbf+2*2;

% Carsinal Spline
ll_spl = sum(log(poisspdf(spike,lambda_spl)));
AIC_spl = -2*ll_spl+2*2;


fprintf('AIC for Linear model %f, \n Quadratic Model %f, \n Gaussian RBf %f, \n Cardinal Spline %f'...
            ,AIC_reg,AIC_quad,AIC_rbf,AIC_spl)

%% Residual

% Linear Regression 
R_reg = cumsum(stats_reg.resid);

%Quadratic Poly. 
R_quad = cumsum(stats_quad.resid);

% Gaussian RBF
R_rbf = cumsum(stats_rbf.resid);

% Cardinal Spline 
R_spl = cumsum(stat_spl.resid);

figure;
hold on 
plot(time,R_reg)
plot(time,R_quad)
plot(time,R_rbf)
plot(time,R_spl)
hold off
title('Residual for Different Models')
xlabel('Time')
ylabel('Residual')
legend('Linear','Quadratic','Gaussian RBF','Cardinal Spline')
grid

