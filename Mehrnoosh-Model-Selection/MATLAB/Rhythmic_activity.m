% Signal processing on EEG data
clear 
close all
clc
%% Loading data
% 
lfp = load('Data/3-2-1-7/eeg.mat');
timee = load('Data/3-2-1-7/time.mat');
lfp = lfp.struct.HPa_03_02_001;
timee = timee.struct.time;
%% Autocovariance

dt = 1/1500; %Sampling intrval
N = length(lfp);
T = N*dt; %Toatal duration of data

[ac,lag] = xcorr(lfp-mean(lfp),500,'biased');
figure;
plot(lag*dt,ac);
xlabel('Lag')
ylabel('Autocovariance')
saveas(gcf,[pwd '/Results/R-3-2-1-7/Autocovariance.png']);
%% Spectrum Analysis

ft = fft(lfp-mean(lfp));
Sxx = (2*dt^2/T)*(ft.*conj(ft));
Sxx = Sxx(1:length(lfp)/2+1);
df = 1/max(T); %Frequancy resolution
fnq = 1/dt/2; %Nyquist Freq. 
f = (0:df:fnq);

%% Spectrogram


sampling_freq = 1/dt;
time_interval = round(sampling_freq);
overlap = round(0.95*sampling_freq);
nfft = round(sampling_freq);
[S,F,T,P] = spectrogram(lfp-mean(lfp),time_interval,overlap,nfft,sampling_freq);
% 
% figure;
% imagesc(T,F,10*log10(P));colormap jet;colorbar;axis xy;
% xlabel('Time[s]');ylabel('Freq.[HZ]');title('Spectrogram of EEG signal')
% 
% saveas(gcf,[pwd '/Results/R-3-2-1-7/Spectrogram.fig']);
% saveas(gcf,[pwd '/Results/R-3-2-1-7/Spectrogram.png']);
%% Theta Rhythm

Wn_th = [4, 10]/fnq;
n_th = 1000;
b_th = fir1(n_th,Wn_th);
lfp_lo_th = filtfilt(b_th,1,lfp);

T = N*dt;
ft_th = fft(lfp_lo_th-mean(lfp_lo_th));
Sxx_th = (2*dt^2/T)*(ft_th.*conj(ft_th));
Sxx_th = Sxx_th(1:length(lfp_lo_th)/2+1);
f = (0:df:fnq);

% Phase and Amplitute
analytic_signal = hilbert(lfp_lo_th);
phi = angle(analytic_signal);
amp = abs(analytic_signal);

figure;
subplot(2,2,1);plot(timee,lfp);hold on;plot(timee,lfp_lo_th);
xlabel('Time[ms]');ylabel('Voltage[V]')
title('Filtered EEG signal (4-10 HZ)')

subplot(2,2,2);plot(f,Sxx,f,Sxx_th)
xlabel('Freq.[Hz]');ylabel('Power [\muV^2/HZ]')
title('Power Spectrum of filtered EEG signal')

subplot(2,2,3);plot(timee,phi);
xlabel('Time');ylabel('Theta Phase')

subplot(2,2,4);envelope(amp,10000,'peak');
xlabel('Time');ylabel('Theta Amplitute')

saveas(gcf,[pwd '/Results/R-3-2-1-7/Theta_Rhythm.fig']);
% saveas(gcf,[pwd '/Results/R-3-2-1-7/Theta_Rhythm.png']);

