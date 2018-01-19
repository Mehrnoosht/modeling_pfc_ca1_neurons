% Signal processing on EEG data
clear 
close all
clc
%% Loading data

eeg = csvread('Data/eeg.csv');
lfp = eeg(:,2);
timee = eeg(:,1);


%% Autocovariance

dt = 1/1500; %Sampling intrval
N = length(lfp);
T = N*dt; %Toatal duration of data

[ac,lag] = xcorr(lfp-mean(lfp),500,'biased');
figure;
plot(lag*dt,ac);
xlabel('Lag')
ylabel('Autocovariance')
%% Spectrum Analysis

ft = fft(lfp-mean(lfp));
Sxx = (2*dt^2/T)*(ft.*conj(ft));
Sxx = Sxx(1:length(lfp)/2+1);
df = 1/max(T); %Frequancy resolution
fnq = 1/dt/2; %Nyquist Freq. 
f = (0:df:fnq);


%% Visualization

figure;
subplot(2,2,1)
plot(timee,lfp);
xlabel('Time[ms]')
ylabel('Voltage[V]')
title('EEG signal')

subplot(2,2,2)
plot(f,Sxx);
xlabel('Freq.[Hz]')
ylabel('Power [\muV^2/HZ]')
title('Power Spectrum of EEG signal')

subplot(2,2,3)
plot(f,10*log10(Sxx/max(Sxx)));
xlabel('Freq.[Hz]')
ylabel('Power [dB]')
title('Power Spectrum of EEG signal')

subplot(2,2,4);
semilogx(f,10*log10(Sxx/max(Sxx)));
xlabel('Freq.[Hz]')
ylabel('Power [dB]')
title('Power Spectrum of EEG signal')
%% Spectrogram

sampling_freq = 1/dt;
time_interval = round(sampling_freq);
overlap = round(0.95*sampling_freq);
nfft = round(sampling_freq);
[S,F,T,P] = spectrogram(lfp-mean(lfp),time_interval,overlap,nfft,sampling_freq);

figure;
imagesc(T,F,10*log10(P));
colormap jet
colorbar;
axis xy;
xlabel('Time[s]');
ylabel('Freq.[HZ]');
title('Spectrogram of EEG signal')
%% Filtered EEG

n=1000; %Filter order
Wn = 17/fnq; %Bandpass
%Building lowpass filter
b = transpose(fir1(n,Wn,'low'));
lfp_low = filtfilt(b,1,lfp);

T = N*dt;
ft_low = fft(lfp_low-mean(lfp_low));
Sxx_low = (2*dt^2/T)*(ft_low.*conj(ft_low));
Sxx_low = Sxx_low(1:length(lfp_low)/2+1);
f = (0:df:fnq);

figure;
subplot(2,1,1)
plot(timee,lfp,timee,lfp_low);
xlabel('Time[ms]')
ylabel('Voltage[V]')
title('Low-pass filtered EEG signal')

subplot(2,1,2);
semilogx(f,10*log10(Sxx/max(Sxx)),f,10*log10(Sxx_low/max(Sxx_low)))
xlim([0 20]);
xlabel('Freq.[Hz]')
ylabel('Power [dB]')
title('Power Spectrum of low-pass filtered EEG signal')
alpha(0.5)
%% Theta Rhythm

Wn_th = [5, 8]/fnq;
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
subplot(2,2,1)
plot(timee,lfp);
hold on
plot(timee,lfp_lo_th);
xlim([4400,4401])
xlabel('Time[ms]')
ylabel('Voltage[V]')
title('Filtered EEG signal (5-8 HZ)')

subplot(2,2,2)
plot(f,Sxx,f,Sxx_th)
xlabel('Freq.[Hz]')
ylabel('Power [\muV^2/HZ]')
title('Power Spectrum of filtered EEG signal')

subplot(2,2,3)
plot(timee,phi);
xlim([4400,4401])
xlabel('Time')
ylabel('Theta Phase')

subplot(2,2,4);
envelope(amp,10000,'peak')
xlabel('Time')
ylabel('Theta Amplitute')


figure;
freqz(analytic_signal,1)