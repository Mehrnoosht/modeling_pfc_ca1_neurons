%% This script is for History dependent model 

 clear;close all;clc 
%% Load data 

lin_pos = load('Data/1-4-4-9/linear_position.mat');
% x_pos = load('Data/1-4-4-9/x_pos.mat');
% y_pos = load('Data/1-4-4-9/y_pos.mat');
spike = load('Data/1-4-4-9/spike.mat');
directionn = load('Data/1-4-4-9/direction.mat');
speed = load('Data/1-4-4-9/speed.mat');
time = load('Data/1-4-4-9/time.mat');
%phi = load('Data/1-4-4-9/phi.mat');

time = time.struct.time;
lin_pos = lin_pos.struct.linear_distance';
% x_pos = x_pos.struct.x_position';
% y_pos = y_pos.struct.y_position';
speed = speed.struct.speed';
directionn = directionn.struct.head_direction';
spike = spike.struct.is_spike';
%phi = phi.phi';

lin_pos(isnan(lin_pos))=0;speed(isnan(speed))=0;
% x_pos(isnan(x_pos))=0;
% y_pos(isnan(y_pos))=0; 
directionn(isnan(directionn))=0;spike = double(spike);

% vx = speed.*cos(directionn);
% vy = speed.*sin(directionn);
%% Partitioning Data
figure; subplot(2,1,1)
plot(x_pos(spike==1),vx(spike==1),'.','MarkerSize',12);xlabel('x_pos')
ylabel('vx')
subplot(2,1,2)
plot(y_pos(spike==1),vy(spike==1),'.','MarkerSize',12);xlabel('y_pos')
ylabel('vy')
saveas(gcf,[pwd '/Results/R-1-4-4-9/vy_ypos.png']);

thsh = 5; 
%% Phase mdoel

[mu_x,mu_y] = meshgrid(linspace(min(lin_pos),max(lin_pos),5),linspace(min(phi),max(phi),5));
variance = 20;
pos_spike = lin_pos(spike==1);
phase_spike = phi(spike==1)';
pos_phi_spike = [pos_spike,phase_spike];
rbf_spike = gaussian_rbf_2d( pos_phi_spike,mu_x,mu_y,variance );

pos_phi = [lin_pos,phi'];
rbf = gaussian_rbf_2d( pos_phi,mu_x,mu_y,variance );
%% History Component
T = length(time);
lag = 200;
Hist = zeros(length(spike)-lag,lag);
for i=1:lag
   Hist(:,i) = spike(lag-i+1:end-i);
end
y = spike(lag+1:end);
%% GLM  for History Component 

c_pt = [-10 0 20 40 60 80 100 120 140 160 180 200 220];
num_c_pts = length(c_pt);s = 0.4; 
X_spl = zeros(lag,length(c_pt));
for i=1:lag
    nearest_c_pt_index = max(find(c_pt<i));
    nearest_c_pt_time = c_pt(nearest_c_pt_index);
    next_c_pt_time = c_pt(nearest_c_pt_index+1);
    u = (i-nearest_c_pt_time)./(next_c_pt_time-nearest_c_pt_time);
    p=[u^3 u^2 u 1]*[-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];
    X_spl(i,nearest_c_pt_index-1:nearest_c_pt_index+2) = p; 
end
X_hist_spl = Hist*X_spl;
[b_spl dev_spl stats_spl] = glmfit(X_hist_spl,y,'poisson');
one = ones(length(y),1);
X_spl_hist = [one X_hist_spl];
lambda_spl = exp(X_spl_hist*b_spl);
%% Visualization of spline basis functions for history
figure;
plot(1:lag,exp(X_spl*b_spl(2:end)))
xlabel('Lags');title('Spline basis function for History');grid
% saveas(gcf,[pwd '/Results/R-1-4-4-9/Spline_Module_history.png']);
%% GLM (Linearized Position)

c_pt_pos = [-10,-5,0,10,20,30,40,50,55,60,70,80,90,110,120,140,170,190,200];
num_c_pts = length(c_pt_pos);s = 0.4; 
spl_pos = CardinalSpline(lin_pos,c_pt_pos,num_c_pts,s);
[b_pos,dev_pos,stat_pos] = glmfit(spl_pos(speed>thsh,:), spike(speed>thsh),'poisson','constant','off');
[yhat,ylo,yhi] = glmval(b_pos,spl_pos(speed>thsh,:),'log',stat_pos,'constant','off');
ran_pos = rank(spl_pos)
cond(spl_pos)

% Perfect Prediction

sum_pp_pos = [];
for j=1:size(spl_pos,2)
    
   idx_pp_pos = find(spl_pos(:,j)~=0);
   sum_pp_pos(j) = sum(spike(idx_pp_pos));
     
end


% Dependecy between columns

imagesc(spl_pos)
idx = [1,2,18,19];
spl_pos_new = spl_pos;
spl_pos_new(:,idx)=[];
rank(spl_pos_new)
imagesc(spl_pos_new)

[b_pos_n,dev_pos_n,stat_pos_n] = glmfit(spl_pos_new(speed>thsh,:), spike(speed>thsh),'poisson','constant','off');

% Missscale
norm_pos = normc(spl_pos);

% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based spline on lin-pos for V>thsh[cm/sec]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_Pos.fig']);

% figure;subplot(2,1,1);
% plot(lin_pos(speed>thsh),yhat);xlabel('Lin-Pos[cm]');ylabel('Firing Rate[spike/sec]')
% subplot(2,1,2);plot(time,lin_pos);xlim([4480,4520]);
% xlabel('Time[sec]');ylabel('Lin-Pos[cm]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/Firing rate-pos-time.fig']);
% saveas(gcf,[pwd '/Results/R-1-4-4-9/Firing rate-pos-time.png']);




%% GLM (Position , Direction)

direction = diff(lin_pos);
direction(direction>=0)=1; % Moving out
direction(direction<0)=0; % Moving in
speedd = speed(2:end);spikee = spike(2:end);tiime = time(2:end);
design_matrix_posdir = [spl_pos(2:end,:) , spl_pos(2:end,:).*direction];
[b_posdir,dev_posdir,stat_posdir] = glmfit(design_matrix_posdir(speedd>thsh,:), spikee(speedd>thsh),'poisson');
[yhat_posdir,ylo_posdir,yhi_posdir] = glmval(b_posdir,design_matrix_posdir(speedd>thsh,:),'log',stat_posdir);


% Perfect Prediction

sum_pp_posdir = [];
for j=1:size(design_matrix_posdir,2)
    
   idx_pp_posdir = find(design_matrix_posdir(:,j)~=0);
   sum_pp_posdir(j) = sum(spike(idx_pp_posdir));
     
end


% Dependecy between columns
imagesc(design_matrix_posdir)
idx =[1,2,18,19,20,21,22,37,38];
design_matrix_posdir_new = design_matrix_posdir;
design_matrix_posdir_new(:,idx)=[];
rank(design_matrix_posdir_new)
imagesc(design_matrix_posdir_new)
[b_posdir_n,dev_posdir_n,stat_posdir_n] = glmfit(design_matrix_posdir_new(speedd>thsh,:), spikee(speedd>thsh),'poisson');

% Missscale
norm_posdir = normc(design_matrix_posdir);

% [b_posdir_reg,fitinfo_posdir] = lassoglm(design_matrix_posdir(speedd>thsh,:), spikee(speedd>thsh),'poisson','CV',3);
% idxLambdaMinDeviance = fitinfo_posdir.IndexMinDeviance;
% B0 = fitinfo_posdir.Intercept(idxLambdaMinDeviance);
% b_pos_dir = [B0; b_posdir_reg(:,idxLambdaMinDeviance)];
% dev_posdir = fitinfo_posdir.Deviance(idxLambdaMinDeviance);

% figure;
% plot(time(speedd>thsh),spikee(speedd>thsh),tiime(speedd>thsh),yhat_posdir*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based on lin-pos and direction for V>thsh[cm/s]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_PosDir.fig']);
%% GLM (Position , speed)

design_matrix_posspd = [spl_pos , speed , spl_pos.*speed];
[b_posspd,dev_posspd,stat_posspd] = glmfit(design_matrix_posspd(speed>thsh,:), spike(speed>thsh),'poisson');
[yhat_posspd,ylo_posspd,yhi_posspd] = glmval(b_posspd,design_matrix_posspd(speed>thsh,:),'log',stat_posspd);
rank_posspd = rank(design_matrix_posspd)

% Perfect Prediction
sum_pp_posspd = [];
for j=1:size(design_matrix_posspd,2)
   idx_pp_posspd = find(design_matrix_posspd(:,j)~=0);
   sum_pp_posspd(j) = sum(spike(idx_pp_posspd));
end

% Dependecy between columns
imagesc(design_matrix_posspd)
idx =[1,2,18,19,20,21,37,38,39];
design_matrix_posspd_new = design_matrix_posspd;
design_matrix_posspd_new(:,idx)=[];
rank(design_matrix_posspd_new)
imagesc(design_matrix_posspd_new)
[b_posspd_n,dev_posspd_n,stat_posspd_n] = glmfit(design_matrix_posspd_new(speedd>thsh,:), spikee(speedd>thsh),'poisson');


% Missscale
norm_posspd = normc(design_matrix_posspd);

% [b_posspd_reg,fitinfo_posspd] = lassoglm(design_matrix_posspd(speed>thsh,:), spike(speed>thsh),'poisson','CV',3);
% idxLambdaMinDeviance = fitinfo_posspd.IndexMinDeviance;
% B0 = fitinfo_posspd.Intercept(idxLambdaMinDeviance);
% b_posspd = [B0; b_posspd_reg(:,idxLambdaMinDeviance)];
% dev_posspd = fitinfo_posspd.Deviance(idxLambdaMinDeviance);


%fprintf('Difference in Dev %f',dev_pos-dev_posspd)
% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat_posspd*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/msec]')
% title('Firing Rate vs. GLM based on lin-pos and speed for V>thsh[cm/s]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_PosSpd.fig']);
%% GLM (Position,Phase)

design_matrix_posphi = [spl_pos , rbf];
[b_posphi,dev_posphi,stat_posphi] = glmfit(design_matrix_posphi(speed>thsh,:), spike(speed>thsh),'poisson');
[yhat_posphi,ylo_posphi,yhi_posphi] = glmval(b_posphi,design_matrix_posphi(speed>thsh,:),'log',stat_posphi);



% [b_posphi_reg,fitinfo_posphi] = lassoglm(design_matrix_posphi(speed>thsh,:), spike(speed>thsh),'poisson','CV',3);
% idxLambdaMinDeviance = fitinfo_posphi.IndexMinDeviance;
% B0 = fitinfo_posphi.Intercept(idxLambdaMinDeviance);
% b_posphi = [B0; b_posphi_reg(:,idxLambdaMinDeviance)];
% dev_posphi = fitinfo_posphi.Deviance(idxLambdaMinDeviance);
%fprintf('Difference in Dev %f',dev_pos-dev_posphi)
% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat_posphi*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based on lin-pos and Theta percession for V>thsh[cm/s]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_PosPhi.fig']);
%% GLM (Position,Phase,Direction)

design_matrix_posdirphi = [spl_pos(2:end,:) , spl_pos(2:end,:).*direction,rbf(2:end,:)];
[b_posdirphi,dev_posdirphi,stat_posdirphi] = glmfit(design_matrix_posdirphi(speedd>thsh,:), spikee(speedd>thsh),'poisson');
[yhat_posdirphi,ylo_posdirphi,yhi_posdirphi] = glmval(b_posdirphi,design_matrix_posdirphi(speedd>thsh,:),'log',stat_posdirphi);

fprintf('Difference in Dev %f',dev_pos-dev_posdirphi)
% figure;
% plot(time(speedd>thsh),spikee(speedd>thsh),tiime(speedd>thsh),yhat_posdir*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based on lin-pos and direction for V>thsh[cm/s]')
%% GLM (Position , Speed, Direction)

spl_posspd = spl_pos.*speed;
design_matrix_posdirspd = [design_matrix_posdir , speedd, spl_posspd(2:end,:)];
[b_posdirspd,dev_posdirspd,stat_posdirspd] = glmfit(design_matrix_posdirspd(speedd>thsh,:), spikee(speedd>thsh),'poisson');
[yhat_posdirspd,ylo_posdirspd,yhi_posdirspd] = glmval(b_posdirspd,design_matrix_posdirspd(speedd>thsh,:),'log',stat_posdirspd);
fprintf('Difference in Dev %f',dev_pos-dev_posdirspd)
% figure;
% plot(tiime(speedd>thsh),spikee(speedd>thsh),tiime(speedd>thsh),yhat_posdirspd*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/msec]')
% title('Firing Rate vs. GLM based on lin-pos and speed and direction for V>thsh[cm/s]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_PosDirSpd.fig']);


%% GLM (Position)

c_pt_xpos = [-10,-5,0,50,55,60,65,70,80,90,100,105,110,120,146,150];
c_pt_ypos = [-8,-5,0,10,20,30,40,45,50,60,70,80,90,110,116,120];

num_c_pts = length(c_pt_xpos);s = 0.4; 
spl_xpos = CardinalSpline(x_pos,c_pt_xpos,num_c_pts,s);
spl_ypos = CardinalSpline(y_pos,c_pt_ypos,num_c_pts,s);
design_matrix_xpos = [spl_xpos,spl_ypos];
[b_xpos,dev_xpos,stat_xpos] = glmfit(design_matrix(speed>thsh,:), spike(speed>thsh),'poisson','constant','off');
[yhat_x,~,~] = glmval(b_xpos,spl_xpos(speed>thsh,:),'log',stat_xpos,'constant','off');

% [b_pos_reg,fitinfo_pos] = lassoglm(spl_pos(speed>thsh,:), spike(speed>thsh),'poisson','CV',3);
% idxLambdaMinDeviance = fitinfo_pos.IndexMinDeviance;
% B0 = fitinfo_pos.Intercept(idxLambdaMinDeviance);
% b_pos = [B0; b_pos_reg(:,idxLambdaMinDeviance)];
% dev_pos = fitinfo_pos.Deviance(idxLambdaMinDeviance);

% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based spline on lin-pos for V>thsh[cm/sec]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_Pos.fig']);
% figure;subplot(2,1,1);
% plot(lin_pos(speed>thsh),yhat);xlabel('Lin-Pos[cm]');ylabel('Firing Rate[spike/sec]')
% subplot(2,1,2);plot(time,lin_pos);xlim([4480,4520]);
% xlabel('Time[sec]');ylabel('Lin-Pos[cm]')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/Firing rate-pos-time.fig']);
% saveas(gcf,[pwd '/Results/R-1-4-4-9/Firing rate-pos-time.png']);

%% GLM (Position , Velocity)


design_matrix_posvel = [design_matrix_xpos , vx , vy];
[b_posvel,dev_posvel,stat_posvel] = glmfit(design_matrix_posvel, spike,'poisson');
[yhat_posvel,ylo_posvel,yhi_posvel] = glmval(b_posvel,design_matrix_posvel,'log',stat_posvel);
lambda_posvel = yhat_posvel(speed>=thsh);
% 
% fprintf('Difference in Dev %f',dev_pos-dev_posvel)
% figure;
% plot(time,spike,time,yhat_posvel*500)
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based on lin-pos and velocity')

%% GLM (Position,History)

spdd = speed(lag+1:end);
tim = time(lag+1:end);
design_matrix_PosHist = [spl_pos(lag+1:end,:) X_hist_spl];
[b_PosHist,dev_PosHist,stat_PosHist] = glmfit(design_matrix_PosHist(spdd>thsh,:), y(spdd>thsh),'poisson');
[yhat_PosHist,ylo_PosHist,yhi_PosHist] = glmval(b_PosHist,design_matrix_PosHist(spdd>thsh,:),'log',stat_PosHist);
% fprintf('Difference in Dev %f',dev_pos-dev_PosHist)

% Perfect Prediction
sum_pp_PosHist = [];
for j=1:size(design_matrix_PosHist,2)
   idx_pp_PosHist = find(design_matrix_PosHist(:,j)~=0);
   sum_pp_PosHist(j) = sum(spike(idx_pp_PosHist));
end

% Dependecy between columns
imagesc(design_matrix_PosHist)
idx =[1,2,18,19,20,31,32];
design_matrix_PosHist_new = design_matrix_PosHist;
design_matrix_PosHist_new(:,idx)=[];
rank(design_matrix_PosHist_new)
imagesc(design_matrix_PosHist_new)


% Missscale
norm_posspd = normc(design_matrix_PosHist);


% figure;
% plot(tim(spdd>thsh),y(spdd>thsh),tim(spdd>thsh),yhat_PosHist*10);
% title('Firing Rate vs. GLM based on lin-pos and history')
% xlabel('Time[sec]');ylabel('Rate of Spiking')
% saveas(gcf,[pwd '/Results/R-1-4-4-9/spl_module_PosHist.fig']);
%% GLM (Position,Direction,History)

direction = diff(lin_pos);
direction(direction>=0)=1; % Moving out
direction(direction<0)=0; % Moving in
spddd = spdd(2:end);
yy = y(2:end);
spl_posdir = spl_pos(2:end,:).*direction;
design_matrix_PosDirHist = [design_matrix_PosHist(2:end,:) , spl_posdir(lag+1:end,:) ];
% [b_PosDirHist,dev_PosDirHist,stat_PosDirHist] = glmfit(design_matrix_PosDirHist(spddd>thsh,:), yy(spddd>thsh),'poisson');
% [yhat_PosDirHist,ylo_PosDirHist,yhi_PosDirHist] = glmval(b_PosDirHist,design_matrix_PosDirHist(spddd>thsh,:),'log',stat_PosDirHist);

% [b_PosDirHist_reg,fitinfo_PosDirHist] = lassoglm(design_matrix_PosDirHist(spdd>thsh,:), y(spdd>thsh),'poisson','CV',3);
% idxLambdaMinDeviance = fitinfo_PosDirHist.IndexMinDeviance;
% B0 = fitinfo_PosDirHist.Intercept(idxLambdaMinDeviance);
% b_PosDirHist = [B0; b_PosDirHist_reg(:,idxLambdaMinDeviance)];
% dev_PosDirHist = fitinfo_PosDirHist.Deviance(idxLambdaMinDeviance);
% yhat_PosDirHist = glmval(b_PosDirHist,design_matrix_PosDirHist(spdd>thsh,:),'log');

fprintf('Difference in Dev %f',dev_PosHist-dev_PosDirHist)
times = time(lag+2:end);
%% GLM (Position,Direction,Speed,History)

spl_posspd = spl_pos.*speed;
design_matrix_PosDirSpdHist = [design_matrix_PosDirHist ,spddd, spl_posspd(lag+2:end,:)];
% [b_PosDirSpdHist,dev_PosDirSpdHist,stat_PosDirSpdHist] = glmfit(design_matrix_PosDirSpdHist(spddd>thsh,:), yy(spddd>thsh),'poisson');
% [yhat_PosDirSpdHist,ylo_PosDirSpdHist,yhi_PosDirSpdHist] = glmval(b_PosDirSpdHist,design_matrix_PosDirSpdHist(spddd>thsh,:),'log',stat_PosDirSpdHist);

% [b_PosDirSpdHist_reg,fitinfo_PosDirSpdHist] = lassoglm(design_matrix_PosDirSpdHist(spddd>thsh,:), yy(spddd>thsh),'poisson','CV',3);
% idxLambdaMinDeviance = fitinfo_PosDirSpdHist.IndexMinDeviance;
% B0 = fitinfo_PosDirSpdHist.Intercept(idxLambdaMinDeviance);
% b_PosDirSpdHist = [B0; b_PosDirSpdHist_reg(:,idxLambdaMinDeviance)];
% dev_PosDirSpdHist = fitinfo_PosDirSpdHist.Deviance(idxLambdaMinDeviance);
% yhat_PosDirSpdHist = glmval(b_PosDirSpdHist,design_matrix_PosDirSpdHist(spdd>thsh,:),'log');

% fprintf('Difference in Dev %f',dev_PosDirHist-dev_PosDirSpdHist)

%% GLM (Position,Direction,Speed,History,Theta Rhythm)

sin_phi = sin(phi(lag+2:end)'); cos_phi = cos(phi(lag+2:end)');
design_matrix_PosDirSpdHistTht = [design_matrix_PosDirSpdHist , cos_phi, sin_phi];
[b_PosDirSpdHistTht,dev_PosDirSpdHistTht,stat_PosDirSpdHistTht] = glmfit(design_matrix_PosDirSpdHistTht(spddd>thsh,:), yy(spddd>thsh),'poisson');
[yhat_PosDirSpdHistTht,ylo_PosDirSpdHistTht,yhi_PosDirSpdHistTht] = glmval(b_PosDirSpdHistTht,design_matrix_PosDirSpdHistTht(spddd>thsh,:),'log',stat_PosDirSpdHistTht);
% 
% fprintf('Difference in Dev %f',dev_PosDirSpdHist-dev_PosDirSpdHistTht)

%% GLM (Position,Direction,Speed,History,Theta)

design_matrix_PosDirSpdHistThtp = [design_matrix_PosDirSpdHist, rbf(lag+2:end,:)];

% Dependecy between columns
imagesc(design_matrix_PosDirSpdHistThtp)
idx =[1,2,18,19,20,31,32];
design_matrix_PosHist_new = design_matrix_PosHist;
design_matrix_PosHist_new(:,idx)=[];
rank(design_matrix_PosHist_new)
imagesc(design_matrix_PosHist_new)


[b_PosDirSpdHistThtp,dev_PosDirSpdHistThtp,stat_PosDirSpdHistThtp] = glmfit(design_matrix_PosDirSpdHistThtp(spddd>thsh,:), yy(spddd>thsh),'poisson');
[yhat_PosDirSpdHistThtp,ylo_PosDirSpdHistThtp,yhi_PosDirSpdHistThtp] = glmval(b_PosDirSpdHistThtp,design_matrix_PosDirSpdHistThtp(spddd>thsh,:),'log',stat_PosDirSpdHistThtp);

% Perfect Prediction
sum_pp_PosDirSpdHistThtp = [];
for j=1:size(design_matrix_PosDirSpdHistThtp,2)
   idx_pp_PosDirSpdHistThtp = find(design_matrix_PosDirSpdHistThtp(:,j)~=0);
   sum_pp_PosDirSpdHistThtp(j) = sum(spike(idx_pp_PosDirSpdHistThtp));
end

% fprintf('Difference in Dev %f',dev_PosDirSpdHist-dev_PosDirSpdHistThtp)
%% KS Plot for History model

figure;
subplot(2,2,1) 
KS_spl2 =  KSplot(yhat_PosHist,y(spdd>thsh));
title('KSPlot-History dependent mdoel ')
xlabel('Model CDF');ylabel('Emperical CDF');

subplot(2,2,2) 
KS_spl2 =  KSplot(yhat_PosDirHist,yy(spddd>thsh));
title('KSPlot-History,Lin-Pos and Direction')
xlabel('Model CDF');ylabel('Emperical CDF')

subplot(2,2,3)
KS_spl =  KSplot(yhat_PosDirSpdHist,yy(spddd>thsh));
title('KSPlot-History,lin-pos,Dir,Speed')
xlabel('Model CDF');ylabel('Emperical CDF');

subplot(2,2,4)
KS_spl =  KSplot(yhat_PosDirSpdHistThtp,yy(spddd>thsh));
title('KSPlot-History,lin-pos,Dir,Speed,Theta Rhythm')
xlabel('Model CDF');ylabel('Emperical CDF');

saveas(gcf,[pwd '/Results/R-1-4-4-9/KSPlot.png']);
saveas(gcf,[pwd '/Results/R-1-4-4-9/KSPlot.fig']);

%% AIC model

% Position and History
ll_PH = sum(log(poisspdf(y(spdd>thsh),yhat_PosHist)));
AIC_PH = -2*ll_PH+2*length(b_PosHist);

% Position,Direction,History 
ll_PDH = sum(log(poisspdf(yy(spddd>thsh),yhat_PosDirHist)));
AIC_PDH = -2*ll_PDH+2*length(b_PosDirHist);

% Position,Direction,Speed,History 
ll_PDSH = sum(log(poisspdf(yy(spddd>thsh),yhat_PosDirSpdHist)));
AIC_PDSH = -2*ll_PDSH+2*length(b_PosDirSpdHist);

% Position,Direction,Speed,Theta Rhythm,History 
ll_PDSHT = sum(log(poisspdf(yy(spddd>thsh),yhat_PosDirSpdHistThtp)));
AIC_PDSHT = -2*ll_PDSHT+2*length(b_PosDirSpdHistThtp);


fprintf('AIC for PH %f, \n PDH %f, \n PDSH %f, \n PDSHT %f'...
            ,AIC_PH,AIC_PDH,AIC_PDSH,AIC_PDSHT)


%% Residual

R_PH = cumsum(stat_PosHist.resid); 
R_PDH = cumsum(stat_PosDirHist.resid);
R_PDSH = cumsum(stat_PosDirSpdHist.resid);
R_PDSHT = cumsum(stat_PosDirSpdHistThtp.resid);

figure;hold on 
timess = time(lag+1:end);
plot(timess(spdd>thsh),R_PH);plot(times(spddd>thsh),R_PDH);plot(times(spddd>thsh),R_PDSH);...
     ,plot(times(spddd>thsh),R_PDSHT)
hold off
title('Residual for Different Models');xlabel('Time');ylabel('Residual')
legend('PH','PDH','PDSH','PDSHT')
grid
saveas(gcf,[pwd '/Results/R-1-4-4-9/Residual_Analysis.png']);
saveas(gcf,[pwd '/Results/R-1-4-4-9/Residual_Analysis.fig']);


%% Emperical Model

N = length(spike);
bins = min(lin_pos):max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = count./occupancy;
linnpos = lin_pos(2:end);
figure;bar(bins,rate);hold on
xlabel('linear Position[cm]');ylabel('Firing Rate[spikes/ms]')
title('Linear Regression GLM')

 