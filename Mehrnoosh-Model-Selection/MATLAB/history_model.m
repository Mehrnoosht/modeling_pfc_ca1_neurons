%% This script is for History dependent model 

%  Load data 

lin_pos = load('Data/3-4-7-2/linear_position.mat');
spike = load('Data/3-4-7-2/spike.mat');
directionn = load('Data/3-4-7-2/direction.mat');
speed = load('Data/3-4-7-2/speed.mat');
time = load('Data/3-4-7-2/time.mat');
% x_pos = load('Data/3-4-7-2/x_pos.mat');
% y_pos = load('Data/3-4-7-2/y_pos.mat');

time = time.struct.time;
lin_pos = lin_pos.struct.linear_distance';
speed = speed.struct.speed';
directionn = directionn.struct.head_direction';
spike = spike.struct.is_spike';
% x_pos = x_pos.struct.x_position';
% y_pos = y_pos.struct.y_position';
phi = phi';

lin_pos(isnan(lin_pos))=0;speed(isnan(speed))=0;
directionn(isnan(directionn))=0;spike = double(spike);
% x_pos(isnan(x_pos))=0;y_pos(isnan(y_pos))=0; 
% vx = speed.*cos(directionn); vy = speed.*sin(directionn);
%% Partitioning Data

% figure; subplot(2,1,1)
% plot(x_pos(spike==1),vx(spike==1),'.','MarkerSize',12);xlabel('x_pos')
% ylabel('vx')
% subplot(2,1,2)
% plot(y_pos(spike==1),vy(spike==1),'.','MarkerSize',12);xlabel('y_pos')
% ylabel('vy')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/vy_ypos.png']);

thsh = 15;
%% Emperical Model 

N = length(spike);
bins = min(lin_pos):10:max(lin_pos);
spikeidx = find(spike==1);
count = hist(lin_pos(spikeidx),bins);
occupancy = hist(lin_pos,bins);
rate = 1500*count./occupancy;

figure; 
bar(bins,rate);xlabel('Linear Position[cm]'); ylabel('Spike/msec')
%% Null Model

design_matrix_nll = ones(size(spike,1),1);
[b_n,dev_n,stat_n] = glmfit(design_matrix_nll(speed>thsh,:),spike(speed>thsh),'poisson','constant','off');
%% GLM  for History Component 

T = length(time);lag = 200;
Hist = zeros(length(spike)-lag,lag);
for i=1:lag
   Hist(:,i) = spike(lag-i+1:end-i);
end
y = spike(lag+1:end);

c_pt = [-10 0 20 40 60 80 100 120 140 160 180 200 220];
num_c_pts = length(c_pt);s = 0.4; 
HistSpl = CdSplHist(lag,c_pt,s);
speedd = speed(lag+1:end);
design_matrix_hist = Hist*HistSpl;
[b_hist dev_hist stat_hist] = glmfit(design_matrix_hist(speedd>thsh,:),y(speedd>thsh),'poisson');
[yhat_hist,ylo_hist,yhi_hist] = glmval(b_hist,design_matrix_hist(speedd>thsh,:),'log',stat_hist);
fprintf('Difference in Dev %f',dev_hist-dev_n)

figure;
plot(1:lag,exp(HistSpl*b_hist(2:end)))
xlabel('Lags');title('Spline basis function for History');grid
saveas(gcf,[pwd '/Results/R-3-4-7-2/Spline_Module_history.png']);
%% GLM (Linearized Position)

c_pt_pos = [-2,-1,0,30,50,70,130,175,190,192];
num_c_pts = length(c_pt_pos);s = 0.4; 
design_matrix_pos = CardinalSpline(lin_pos,c_pt_pos,num_c_pts,s);
ran_pos = rank(design_matrix_pos)

% Perfect Prediction
[design_matrix_pos,pp_p,idx_pp_p] = PrfctPrd(design_matrix_pos,spike);

% Dependecy between columns
imagesc(design_matrix_pos);colorbar;
design_matrix_pos(:,[1,9])=[];
design_matrix_pos(:,[4])=[];
% GLMFit
[b_pos,dev_pos,stat_pos] = glmfit(design_matrix_pos(speed>thsh,:), spike(speed>thsh),'poisson');
[yhat_pos,ylo_pos,yhi_pos] = glmval(b_pos,design_matrix_pos(speed>thsh,:),'log',stat_pos);
fprintf('Difference in Dev %f',dev_n-dev_pos)

del_dev = [];
for i=1:size(design_matrix_pos,2)
    
    [b_pos1,dev_pos1,stat_pos1] = glmfit(design_matrix_pos(speed>thsh,1:i), spike(speed>thsh),'poisson');
     [b_pos2,dev_pos2,stat_pos2] = glmfit(design_matrix_pos(speed>thsh,1:i+1), spike(speed>thsh),'poisson');
    del_dev = [del_dev dev_pos1-dev_pos2];
end

% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat_pos*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_Pos1.fig']);
figure;
plot(lin_pos(speed>thsh),spike(speed>thsh),lin_pos(speed>thsh),yhat_pos*1500,'.');
xlabel('Lin-Pos[cm]');ylabel('Firing Rate[spike/sec]')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/Firing rate-pos-time.fig']);
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_Pos2.png']);
%% GLM (Position , Direction)

direction = diff(lin_pos);
direction = [direction<0 , direction>=0];
spd = speed(2:end);spikee = spike(2:end);tiime = time(2:end);
design_matrix_posdir = [design_matrix_pos(2:end,:).*direction(:,1),design_matrix_pos(2:end,:).*direction(:,2)];
rank(design_matrix_posdir)

% Perfect Prediction
[design_matrix_posdir_new,pp_pd,idx_pp_pd] = PrfctPrd(design_matrix_posdir,spikee);
% Dependecy between columns
imagesc(design_matrix_posdir_new);colorbar;

%Misscaling
a = vecnorm(design_matrix_posdir)./size(design_matrix_posdir,1);

%GLMFit
[b_posdir,dev_posdir,stat_posdir] = glmfit(design_matrix_posdir_new(spd>thsh,:), spikee(spd>thsh),'poisson');
[yhat_posdir,ylo_posdir,yhi_posdir] = glmval(b_posdir,design_matrix_posdir_new(spd>thsh,:),'log',stat_posdir);
yhat_posdir = yhat_posdir.*exp(-100*design_matrix_posdir(spd>thsh,idx_pp_pd));
dev_posdir =-2*(nansum(log(poisspdf(spikee(spd>thsh),yhat_posdir)))- ...
    nansum(log(poisspdf(spikee(spd>thsh),spikee(spd>thsh)))));
fprintf('Difference in Dev %f',dev_pos-dev_posdir)

% figure;
% plot(time(speedd>thsh),spikee(speedd>thsh),tiime(speedd>thsh),yhat_posdir*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based on lin-pos and direction for V>thsh[cm/s]')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosDir1.fig']);
figure;hold on 
dir = direction(spd>thsh,:);
plot(lin_pos(spd>thsh & direction(:,1)==1),spike(spd>thsh & direction(:,1)==1),...
   lin_pos(spd>thsh & direction(:,2)==1),spike(spd>thsh & direction(:,2)==1));
plot(lin_pos(spd>thsh &  direction(:,1)==1),yhat_posdir(dir(:,1)==1)*1500,'.')
plot(lin_pos(spd>thsh &  direction(:,2)==1),yhat_posdir(dir(:,2)==1)*1500,'.'); hold off
xlabel('Lin-Pos[cm]');ylabel('Firing Rate[spike/sec]');
legend('Spike-Inbound','Spike-Outbound','FiringRate-Inbound','FiringRate-Outbound')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosDir2.fig']);
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosDir2.png']);
%% GLM (Position , speed)

design_matrix_posspd = [design_matrix_pos.*speed];
rank(design_matrix_posspd)

% Perfect Prediction
design_matrix_posspd = PrfctPrd(design_matrix_posspd,spike);

% Dependecy between columns
imagesc(design_matrix_posspd);colorbar;

% GLMFit
[b_posspd,dev_posspd,stat_posspd] = glmfit(design_matrix_posspd(speed>thsh,:), spike(speed>thsh),'poisson');
[yhat_posspd,ylo_posspd,yhi_posspd] = glmval(b_posspd,design_matrix_posspd(speed>thsh,:),'log',stat_posspd);
fprintf('Difference in Dev %f',dev_pos-dev_posspd)

% 
% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat_posspd*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/msec]')
% title('Firing Rate vs. GLM based on lin-pos and speed for V>thsh[cm/s]')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosSpd.fig']);
%% Phase mdoel

% Gaussian RBF 
[mu_x,mu_y] = meshgrid(linspace(min(lin_pos),max(lin_pos),5),linspace(min(phi),max(phi),5));
variance = 20;pos_spike = lin_pos(spike==1);phase_spike = phi(spike==1);
pos_phi_spike = [pos_spike,phase_spike];
rbf_spike = gaussian_rbf_2d( pos_phi_spike,mu_x,mu_y,variance );
pos_phi = [lin_pos,phi];
rbf = gaussian_rbf_2d( pos_phi,mu_x,mu_y,variance );

% Indicator Fundtions
ind_phase = [];nn=5;
phi_prt = linspace(min(phi),max(phi)+0.01,nn);
for i=1:nn-1
   ind_phase(:,i) = [phi>=phi_prt(i) & phi<phi_prt(i+1)];   
end
%% GLM (Position,Phase)

design_matrix_posphi = [];
for i=1:nn-1
   design_matrix_posphi = [design_matrix_posphi,design_matrix_pos.*ind_phase(:,i)];
end
rank(design_matrix_posphi)

% Perfect Prediction
[design_matrix_posphi,pp_pp,idx_pp_pp] = PrfctPrd(design_matrix_posphi,spike);

% Dependecy between columns
imagesc(design_matrix_posphi);colorbar;

% GLMFit
[b_posphi,dev_posphi,stat_posphi] = glmfit(design_matrix_posphi(speed>thsh,:), spike(speed>thsh),'poisson');
[yhat_posphi,ylo_posphi,yhi_posphi] = glmval(b_posphi,design_matrix_posphi(speed>thsh,:),'log',stat_posphi);
fprintf('Difference in Dev %f',dev_pos-dev_posphi)

% figure;
% plot(time(speed>thsh),spike(speed>thsh),time(speed>thsh),yhat_posphi*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
% title('Firing Rate vs. GLM based on lin-pos and Theta percession for V>thsh[cm/s]')
figure;
ind_phi = ind_phase(speed>thsh,:);
subplot(2,2,1);
plot(lin_pos(speed>thsh & ind_phase(:,1)==1),spike(speed>thsh & ind_phase(:,1)==1),...
     lin_pos(speed>thsh & ind_phase(:,1)==1),yhat_posphi(ind_phi(:,1)==1)*1500,'.');
subplot(2,2,2);
plot(lin_pos(speed>thsh & ind_phase(:,2)==1),spike(speed>thsh & ind_phase(:,2)==1),...
     lin_pos(speed>thsh & ind_phase(:,2)==1),yhat_posphi(ind_phi(:,2)==1)*1500,'.'); 
subplot(2,2,3);
plot(lin_pos(speed>thsh & ind_phase(:,3)==1),spike(speed>thsh & ind_phase(:,3)==1),...
     lin_pos(speed>thsh & ind_phase(:,3)==1),yhat_posphi(ind_phi(:,3)==1)*1500,'.'); 
subplot(2,2,4);
plot(lin_pos(speed>thsh & ind_phase(:,4)==1),spike(speed>thsh & ind_phase(:,4)==1),...
     lin_pos(speed>thsh & ind_phase(:,4)==1),yhat_posphi(ind_phi(:,4)==1)*1500,'.');  
xlabel('Lin-Pos[cm]');ylabel('Firing Rate[spike/sec]')
% saveas(gcf,[pwd '/Results/R-3-4-7-2/Firing rate-pos-time.fig']);
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosPhi1.png']);
figure;
len = length(phi(spike==1 & speed>thsh));
scatter3(lin_pos(spike==1 & speed>thsh),phi(spike==1 & speed>thsh),ones(len,1));hold on;
plot3(lin_pos(speed>thsh),phi(speed>thsh),yhat_posphi*1500,'.');
xlabel('Linear Position[cm]');ylabel('Firing rate [spike/sec]');
legend('Spike','Friringn rate')
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosPhi2.fig']);

figure;
len = length(phi(spike==1 & speed>thsh));
ct = [-10,-5,10,40,100,160,188,190];num_ct = length(ct);
pos = linspace(min(lin_pos),max(lin_pos),100);
phai = linspace(min(phi),max(phi),100);
[Pos,Phai] = meshgrid(pos,phai);One = ones(size(Pos));
a = CardinalSpline(pos,ct,num_ct,s);

ind_phai = [];nn=5;
phai_prt = linspace(-pi,pi+0.01,nn);
for i=1:nn-1
   ind_phai(:,i) = [phai>=phai_prt(i) & phai<phai_prt(i+1)];   
end
c = zeros(100,100,24);
k=1;
for j=1:4
   for i=1:6
     c(:,:,k) = b_posphi(k+1).*(a(:,i).*ind_phai(:,j)).*ones(100,100);
     k=k+1;
   end
end
lambda_phipos = 1500*exp(b_posphi(1).*One+sum(c,3));
surf(Pos,Phai,lambda_phipos)
xlabel('Linear Position[cm]');ylabel('Phase');zlabel('Firing rate [spike/sec]')
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosPhi3.fig']);
%% GLM (Position,Phase,Direction)

design_matrix_posphidir = [design_matrix_posphi(2:end,:).*direction(:,1),design_matrix_posphi(2:end,:).*direction(:,2)];
rank(design_matrix_posphidir)

% Perfect Prediction
[design_matrix_posphidir_new,pp_ppd,idx_pp_ppd] = PrfctPrd(design_matrix_posphidir,spikee);

% Dependecy between columns
imagesc(design_matrix_posphidir);colorbar;
rank(design_matrix_posphidir_new)


[b_posphidir,dev_posphidir,stat_posphidir] = glmfit(design_matrix_posphidir_new(spd>thsh,:), spikee(spd>thsh),'poisson');
[yhat_posphidir,ylo_posphidir,yhi_posphidir] = glmval(b_posphidir,design_matrix_posphidir_new(spd>thsh,:),'log',stat_posphidir);
design_matrix_posphidir_n = [design_matrix_posphidir_new, design_matrix_posphidir(:,idx_pp_ppd)];
b_posphidir_n = [b_posphidir',-100*ones(1,length(idx_pp_ppd))]';
yhat_posphidir = glmval(b_posphidir_n,design_matrix_posphidir_n(spd>thsh,:),'log');
dev_posphidir =-2*(nansum(log(poisspdf(spikee(spd>thsh),yhat_posphidir)))- ...
    sum(log(poisspdf(spikee(spd>thsh),spikee(spd>thsh)))));

fprintf('Difference in Dev %f',dev_pos-dev_posphidir)

%Misscaling
a = vecnorm(design_matrix_posphidir)./size(design_matrix_posphidir,1);

% figure;
% plot(time(speedd>thsh),spikee(speedd>thsh),tiime(speedd>thsh),yhat_posphidir*1500,'.')
% xlabel('time[S]');ylabel('Rate of Spiking[Spike/sec]')
figure;
ind_phi = ind_phase(speed>thsh,:);dir = direction(spd>thsh,:);ind_ph=ind_phase(2:end,:);
subplot(2,2,1);
plot(lin_pos(spd>thsh & ind_ph(:,1)==1 & direction(:,1)==1),spike(spd>thsh & ind_ph(:,1)==1 & direction(:,1)==1),...
     lin_pos(spd>thsh & ind_ph(:,1)==1 & direction(:,1)==1),yhat_posphidir(ind_phi(:,1)==1 & dir(:,1)==1)*1500,'.');hold on 
plot(lin_pos(spd>thsh & ind_ph(:,1)==1 & direction(:,2)==1),spike(spd>thsh & ind_ph(:,1)==1 & direction(:,2)==1),...
     lin_pos(spd>thsh & ind_ph(:,1)==1 & direction(:,2)==1),yhat_posphidir(ind_phi(:,1)==1 & dir(:,2)==1)*1500,'.') 
xlabel('Lin-Pos[cm]');ylabel('Firing Rate[spike/sec]') 
legend('Spk-Inb','Spk-Outb','FrRt-Inb','FrRt-Outb')
subplot(2,2,2);
plot(lin_pos(spd>thsh & ind_ph(:,2)==1 & direction(:,1)==1),spike(spd>thsh & ind_ph(:,2)==1 & direction(:,1)==1),...
     lin_pos(spd>thsh & ind_ph(:,2)==1 & direction(:,1)==1),yhat_posphidir(ind_phi(:,2)==1 & dir(:,1)==1)*1500,'.');hold on 
plot(lin_pos(spd>thsh & ind_ph(:,2)==1 & direction(:,2)==1),spike(spd>thsh & ind_ph(:,2)==1 & direction(:,2)==1),...
     lin_pos(spd>thsh & ind_ph(:,2)==1 & direction(:,2)==1),yhat_posphidir(ind_phi(:,2)==1 & dir(:,2)==1)*1500,'.') 
subplot(2,2,3);
plot(lin_pos(spd>thsh & ind_ph(:,3)==1 & direction(:,1)==1),spike(spd>thsh & ind_ph(:,3)==1 & direction(:,1)==1),...
     lin_pos(spd>thsh & ind_ph(:,3)==1 & direction(:,1)==1),yhat_posphidir(ind_phi(:,3)==1 & dir(:,1)==1)*1500,'.');hold on 
plot(lin_pos(spd>thsh & ind_ph(:,3)==1 & direction(:,2)==1),spike(spd>thsh & ind_ph(:,3)==1 & direction(:,2)==1),...
     lin_pos(spd>thsh & ind_ph(:,3)==1 & direction(:,2)==1),yhat_posphidir(ind_phi(:,3)==1 & dir(:,2)==1)*1500,'.') 
subplot(2,2,4);
plot(lin_pos(spd>thsh & ind_ph(:,4)==1 & direction(:,1)==1),spike(spd>thsh & ind_ph(:,4)==1 & direction(:,1)==1),...
     lin_pos(spd>thsh & ind_ph(:,4)==1 & direction(:,1)==1),yhat_posphidir(ind_phi(:,4)==1 & dir(:,1)==1)*1500,'.');hold on 
plot(lin_pos(spd>thsh & ind_ph(:,4)==1 & direction(:,2)==1),spike(spd>thsh & ind_ph(:,4)==1 & direction(:,2)==1),...
     lin_pos(spd>thsh & ind_ph(:,4)==1 & direction(:,2)==1),yhat_posphidir(ind_phi(:,4)==1 & dir(:,2)==1)*1500,'.') 
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosPhiDir1.fig']);
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosPhiDir1.png']);
figure;
len1 = length(phi(spikee==1 & spd>thsh & direction(:,1)==1));
len2 = length(phi(spikee==1 & spd>thsh & direction(:,2)==1));
scatter3(lin_pos(spikee==1 & spd>thsh & direction(:,1)==1),phi(spikee==1 & spd>thsh & direction(:,1)==1),...
         ones(len1,1));hold on;
scatter3(lin_pos(spikee==1 & spd>thsh & direction(:,2)==1),phi(spikee==1 & spd>thsh & direction(:,2)==1),...
         ones(len2,1));
plot3(lin_pos(spd>thsh & direction(:,1)==1),phi(spd>thsh & direction(:,1)==1),...
      yhat_posphi(dir(:,1)==1)*1500,'.');hold on
plot3(lin_pos(spd>thsh & direction(:,2)==1),phi(spd>thsh & direction(:,2)==1),...
      yhat_posphi(dir(:,2)==1)*1500,'.');
xlabel('Linear Position[cm]');ylabel('Firing rate [spike/sec]');
legend('Spk-Inb','Spk-Outb','FrRt-Inb','FrRt-Outb')
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_PosPhiDir2.fig']);
%% GLM (Position,Phase,Direction,Speed)

design_matrix_posphidirspd = [design_matrix_posphidir.*spd];
rank(design_matrix_posphidirspd)

% Perfect Prediction
[design_matrix_posphidirspd_new,pp_ppds,idx_pp_ppds]  = PrfctPrd(design_matrix_posphidirspd,spikee);

% Dependecy between columns
imagesc(design_matrix_posphidirspd)
rank(design_matrix_posphidirspd)

[b_posphidirspd,dev_posphidirspd,stat_posphidirspd] = glmfit(design_matrix_posphidirspd_new(spd>thsh,:), spikee(spd>thsh),'poisson');
[yhat_posphidirspd,ylo_posphidirspd,yhi_posphidirspd] = glmval(b_posphidirspd,design_matrix_posphidirspd(spd>thsh,:),'log',stat_posphidirspd);
design_matrix_posphidirspd_n = [design_matrix_posphidirspd_new, design_matrix_posphidirspd(:,idx_pp_ppds)];
b_posphidirspd_n = [b_posphidirspd',-10*ones(1,length(idx_pp_ppds))]';
yhat_posphidirspd = glmval(b_posphidirspd_n,design_matrix_posphidirspd_n(spd>thsh,:),'log');
dev_posphidirspd =-2*(nansum(log(poisspdf(spikee(spd>thsh),yhat_posphidirspd)))- ...
    sum(log(poisspdf(spikee(spd>thsh),spikee(spd>thsh)))));
%% GLM (Position,Phase,Direction,History)

spdd = speed(lag+2:end);timm = time(lag+2:end);spk=spike(lag+2:end);
design_matrix_full = [design_matrix_posphidir(lag+1:end,:),design_matrix_hist(2:end,:)];
rank(design_matrix_full)

% Perfect Prediction
[design_matrix_full_new,pp_full,idx_pp_full] = PrfctPrd(design_matrix_full,spk);

% Dependecy between columns
imagesc(design_matrix_full)
rank(design_matrix_full)

[b_full,dev_full,stat_full] = glmfit(design_matrix_full_new(spdd>thsh,:), spk(spdd>thsh),'poisson');
[yhat_full,ylo_full,yhi_full] = glmval(b_full,design_matrix_full_new(spdd>thsh,:),'log',stat_full);
design_matrix_full_n = [design_matrix_full_new, design_matrix_full(:,idx_pp_full)];
b_full_n = [b_full',-100*ones(1,length(idx_pp_full))]';
yhat_full = glmval(b_full_n,design_matrix_full_n(spdd>thsh,:),'log');
dev_full =-2*(nansum(log(poisspdf(spikee(spdd>thsh),yhat_full)))- ...
    sum(log(poisspdf(spikee(spdd>thsh),spikee(spdd>thsh)))));
fprintf('Difference in Dev %f',dev_posphidir-dev_full)
%% Non-Parametric Model 

posspd = lin_pos(speed>thsh);phispd = phi(speed>thsh);
posspdspk = lin_pos(speed>thsh & spike==1);phispdspk = spike(speed>thsh & spike==1);
dataspk = [posspdspk,phispdspk];data = [posspd ,phispd];
mu = [posspd, phispd];sigma = [5,0.5];
len_y = length(data);
y_nprmt = zeros(len_y,1);
for i=1:len_y 
    num = sum(mvnpdf(dataspk,mu(i,:),sigma));
    den = sum(mvnpdf(data,mu(i,:),sigma));
    y_nprmt(i) = num./den;  
end
dev_nprmt =-2*(sum(log(poisspdf(spike(speed>thsh),y_nprmt)))- ...
    nansum(log(poisspdf(spike(speed>thsh),spike(speed>thsh)))));
dev_nprmt-dev_posphi
AIC_nprmt = -2*(sum(log(poisspdf(spike(speed>thsh),y_nprmt))))+2*length(y_nprmt);
figure;
len = length(phi(spike==1 & speed>thsh));
scatter3(lin_pos(spike==1 & speed>thsh),phi(spike==1 & speed>thsh),ones(len,1));hold on;
plot3(lin_pos(speed>thsh),phi(speed>thsh),y_nprmt*1500,'.');
xlabel('Linear Position[cm]');ylabel('Firing rate [spike/sec]');
legend('Spike','Friringn rate')
saveas(gcf,[pwd '/Results/R-3-4-7-2/spl_module_Nonp_PosPhi1.fig']);
figure;
subplot(2,1,1)
KS_spl =  KSplot(y_nprmt,spike(speed>thsh));
title('KS-Non parametric model for pos,phase')
xlabel('Model CDF');ylabel('Emperical CDF');
subplot(2,1,2)
KS_spl =  KSplot(yhat_posphi,spike(speed>thsh));
title('KSPlot lin-pos,phase')
xlabel('Model CDF');ylabel('Emperical CDF')
saveas(gcf,[pwd '/Results/R-3-4-7-2/KS2.fig']);
%% Deviance Analysis

dev_hn = (dev_n-dev_hist)/dev_n
devf_hn = (dev_n-dev_hist)/(dev_n-dev_full)
dev_pn = (dev_n-dev_pos)/dev_n
devf_pn = (dev_n-dev_pos)/(dev_n-dev_full)
dev_pp = (dev_n-dev_posphi)/dev_n
devf_pp = (dev_n-dev_posphi)/(dev_n-dev_full)
dev_pd = (dev_n-dev_posdir)/dev_n
devf_pd = (dev_n-dev_posdir)/(dev_n-dev_full)
dev_ps = (dev_n-dev_posspd)/dev_n
devf_ps = (dev_n-dev_posspd)/(dev_n-dev_full)
dev_nprmtn = (dev_n-dev_nprmt)/dev_n
devf_nprmtn = (dev_n-dev_nprmt)/(dev_n-dev_full)

dev_ppp = (dev_pos-dev_posphi)/dev_pos
devf_ppp = (dev_pos-dev_posphi)/(dev_pos-dev_full)
dev_ppid = (dev_posphi-dev_posphidir)/dev_posphi
devf_ppid = (dev_posphi-dev_posphidir)/(dev_posphi-dev_full)
% dev_ppids = (dev_posphidir-dev_posphidirspd)/dev_posphidir
% devf_ppids = (dev_posphidir-dev_posphidirspd)/(dev_posphidir-dev_dev_full)
dev_full = (dev_posphidir-dev_full)/dev_posphidir
%% AIC model

ll_P = sum(log(poisspdf(spike(speed>thsh),yhat_pos)));
AIC_P = -2*ll_P+2*length(b_pos);

ll_PP = sum(log(poisspdf(spike(speed>thsh),yhat_posphi)));
AIC_PP = -2*ll_PP+2*length(b_posphi);

ll_PPD = sum(log(poisspdf(spikee(spd>thsh),yhat_posphidir)));
AIC_PPD = -2*ll_PPD+2*length(b_posphi);

% ll_PPDS = sum(log(poisspdf(spikee(spd>thsh),yhat_posphidirspd)));
% AIC_PPDS = -2*ll_PPDS+2*length(b_posphidirspd);

ll_full= sum(log(poisspdf(spk(spdd>thsh),yhat_full)));
AIC_full = -2*ll_full+2*length(b_full);
fprintf('AIC for P %f, \n PP %f, \n PPD %f, \n full %f'...
            ,AIC_P,AIC_PP,AIC_PPD,AIC_full)
%% KS Plot

figure;
subplot(2,2,1) 
KS_spl2 =  KSplot(yhat_pos,spike(speed>thsh));
title('lin-pos')
xlabel('Model CDF');ylabel('Emperical CDF');

subplot(2,2,2) 
KS_spl2 =  KSplot(yhat_posphi,spike(speed>thsh));
title('lin-pos,phase')
xlabel('Model CDF');ylabel('Emperical CDF')

subplot(2,2,3)
KS_spl2 =  KSplot(yhat_posphidir,spikee(spd>thsh));
title('lin-pos,phase,dir')
xlabel('Model CDF');ylabel('Emperical CDF')

subplot(2,2,4)
KS_spl =  KSplot(yhat_full,spk(spdd>thsh));
title('lin-pos,dir,spd,hist')
xlabel('Model CDF');ylabel('Emperical CDF');

saveas(gcf,[pwd '/Results/R-3-4-7-2/KSPlot.png']);
saveas(gcf,[pwd '/Results/R-3-4-7-2/KSPlot.fig']);       
%% Residual

R_P = cumsum(stat_pos.resid); 
R_PP = cumsum(stat_posphi.resid);
R_PPD = cumsum(stat_posphidir.resid);
% R_PPDS = cumsum(stat_posphidirspd.resid);
R_full = cumsum(stat_full.resid);

figure;hold on 
times = time(2:end);
plot(time(speed>thsh),R_P);plot(time(speed>thsh),R_PP);plot(times(spd>thsh),R_PPD);...
% plot(times(spd>thsh),R_PPDS);
plot(timm(spdd>thsh),R_full)
hold off
title('Cumulative Residual');xlabel('Time');ylabel('Residual')
% legend('P','PP','PPD','PPDS','PPDSH')
 legend('P','PP','PPD','PPDH')
grid
saveas(gcf,[pwd '/Results/R-3-4-7-2/Residual_Analysis.png']);
% saveas(gcf,[pwd '/Results/R-3-4-7-2/Residual_Analysis.fig']);