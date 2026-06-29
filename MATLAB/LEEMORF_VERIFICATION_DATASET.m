close all; clear all;
set(groot,'defaultAxesFontSize', 14)
set(0, 'DefaultAxesFontName', 'Times');

%% Load file
file = "variable"
% file = "stable"

if file == "stable"
    [x,fs]=audioread('J:\Dyski współdzielone\PKsiazek_Pszczoly\Matlab\Sound_verif_set\stable buzz worker\2\Buzz-00.wav');
else
    [x,fs]=audioread('J:\Dyski współdzielone\PKsiazek_Pszczoly\Matlab\Sound_verif_set\variable buzz drone\1\Buzz_variable-01.wav');
end

% Universal configurations
filter_order = 20;
tau = 50/1000;
fd = 2000; %[Hz]

% Rescale and highpass filter signal
x = rescale(x); 
[B, A] = ellip(3, 0.1, 50, 2*100/fs, 'high');
x = filtfilt(B, A, x); 
y = resample(x, fd, fs);

% Setup time arrays
T=size(x,1); t=(0:T-1)'/fs; N=size(y,1); n=(0:N-1)'/fd;

activity_mask = zeros(length(y), 1);
for winstart = 1:100:(length(y)-100)
    rms_win = rms(y(winstart:winstart+100));
    activity_mask(winstart:winstart+100) = rms_win > 0.01;
end

% ========================================================================
%% 1. LEEMORF Track Processing (With Filtering)
% ========================================================================
lambda = 1 - (1/fd)/((1/fd)+tau);
AR_P = filter_order;
[Ro, ~, ~] = lee_morf_new(y, AR_P, lambda);
A_coef = ktoa(Ro');
SD = fft(A_coef, 2048);
S = 20*log10(1./abs(SD(1:1024, :))); 
f = fd*(0:1023)/2048; 
M = S > prctile(S, 95, "all");
SF = S; SF(~M) = 0;
rdg_leemorf = tfridge(SF(1:300, :), f(1:300), 0);
rdg_leemorf = rdg_leemorf(:);

% Process LEEMORF Mask and Segment
if length(activity_mask) < length(rdg_leemorf)
    mask_leemorf = padarray(activity_mask(:), length(rdg_leemorf) - length(activity_mask), 0, "post");
else
    mask_leemorf = activity_mask(1:length(rdg_leemorf));
end
rdg_leemorf = rdg_leemorf .* mask_leemorf(:);
zeromask = (rdg_leemorf == 0);
segment_ends = find(diff(zeromask, 1) == 1);
segment_starts = find(diff(zeromask, 1) == -1);
if ~isempty(zeromask) && zeromask(1) == 0, segment_starts = [0; segment_starts]; end
if length(segment_ends) < length(segment_starts), segment_ends = [segment_ends; length(rdg_leemorf)]; end
leemorf_segfiltered = zeros(length(rdg_leemorf), 1);
[b2, a2] = ellip(5, 1, 80, 0.1, "low"); 
if ~isempty(segment_starts)
    all_seglens = segment_ends - segment_starts;
    [max_len, max_idx] = max(all_seglens);
    if max_len > 2
        segstart = segment_starts(max_idx); segend = segment_ends(max_idx);
        leemorf_segfiltered(segstart+1:segend) = filtfilt(b2, a2, rdg_leemorf(segstart+1:segend));
    end
end
leemorf_segfiltered(leemorf_segfiltered == 0) = NaN;

% --- Shift LEEMORF 50 ms back to offset causal group delay ---
leemorf_shift_samples = round(0.025 * fd); 
if length(leemorf_segfiltered) > leemorf_shift_samples
    leemorf_segfiltered = [leemorf_segfiltered(leemorf_shift_samples+1:end); NaN(leemorf_shift_samples, 1)];
end

% ========================================================================
%% 2. AUTOCORRELATION Track Processing (Bypass Filtering)
% ========================================================================
[acorrestim, acorrindices] = f0_estimates(y, fd, 100, 300, 50/1000);
track_autocorr = acorrestim(:);
mask_acorr = activity_mask(acorrindices);
track_acorr_masked = track_autocorr .* mask_acorr(:);
zeromask_acorr = (track_acorr_masked == 0);
ends_acorr = find(diff(zeromask_acorr, 1) == 1);
starts_acorr = find(diff(zeromask_acorr, 1) == -1);
if ~isempty(zeromask_acorr) && zeromask_acorr(1) == 0, starts_acorr = [0; starts_acorr]; end
if length(ends_acorr) < length(starts_acorr), ends_acorr = [ends_acorr; length(track_acorr_masked)]; end
acorr_segprocessed = zeros(length(track_acorr_masked), 1);
if ~isempty(starts_acorr)
    all_seglens_acorr = ends_acorr - starts_acorr;
    [max_len_acorr, max_idx_acorr] = max(all_seglens_acorr);
    if max_len_acorr > 2
        segstart = starts_acorr(max_idx_acorr); segend = ends_acorr(max_idx_acorr);
        acorr_segprocessed(segstart+1:segend) = track_acorr_masked(segstart+1:segend);
    end
end
acorr_segprocessed(acorr_segprocessed == 0) = NaN;
n_acorr = (acorrindices - 1) / fd; % Dynamic time mapping axis

% ========================================================================
%% 3. STFT Track Processing (Bypass Filtering)
% ========================================================================
[s_stft, f_stft, t_stft] = stft(y, fd, 'Window', hamming(round(0.05*fd)), 'OverlapLength', round(0.05*fd/2), 'FrequencyRange', 'onesided');
S_stft = 20*log10(abs(s_stft));
f_idx = find(f_stft >= 50 & f_stft <= 300);
track_stft = tfridge(abs(s_stft(f_idx, :)), f_stft(f_idx), 0.1);
track_stft = track_stft(:);
stft_indices = min(max(round(t_stft * fd) + 1, 1), length(activity_mask));
mask_stft = activity_mask(stft_indices);
track_stft_masked = track_stft .* mask_stft(:);
zeromask_stft = (track_stft_masked == 0);
ends_stft = find(diff(zeromask_stft, 1) == 1);
starts_stft = find(diff(zeromask_stft, 1) == -1);
if ~isempty(zeromask_stft) && zeromask_stft(1) == 0, starts_stft = [0; starts_stft]; end
if length(ends_stft) < length(starts_stft), ends_stft = [ends_stft; length(track_stft_masked)]; end
stft_segprocessed = zeros(length(track_stft_masked), 1);
if ~isempty(starts_stft)
    all_seglens_stft = ends_stft - starts_stft;
    [max_len_stft, max_idx_stft] = max(all_seglens_stft);
    if max_len_stft > 2
        segstart = starts_stft(max_idx_stft); segend = ends_stft(max_idx_stft);
        stft_segprocessed(segstart+1:segend) = track_stft_masked(segstart+1:segend);
    end
end
stft_segprocessed(stft_segprocessed == 0) = NaN;

% ========================================================================
%% 4. YIN Track Processing (Bypass Filtering)
% ========================================================================
P.sr = fd;
P.minf0 = 100;
P.maxf0 = 300;
P.hop = 16;
P.thresh = 0.1;
r_yin = yin(y, P);
track_yin = 440 * (2.^r_yin.f0(:)); 
yin_indices = 1 + (0:length(track_yin)-1)' * P.hop;
yin_indices = min(max(yin_indices, 1), length(activity_mask));
mask_yin = activity_mask(yin_indices);
track_yin_masked = track_yin .* mask_yin(:);
zeromask_yin = (track_yin_masked == 0);
ends_yin = find(diff(zeromask_yin, 1) == 1);
starts_yin = find(diff(zeromask_yin, 1) == -1);
if ~isempty(zeromask_yin) && zeromask_yin(1) == 0, starts_yin = [0; starts_yin]; end
if length(ends_yin) < length(starts_yin), ends_yin = [ends_yin; length(track_yin_masked)]; end
yin_segprocessed = zeros(length(track_yin_masked), 1);
if ~isempty(starts_yin)
    all_seglens_yin = ends_yin - starts_yin;
    [max_len_yin, max_idx_yin] = max(all_seglens_yin);
    if max_len_yin > 2
        segstart = starts_yin(max_idx_yin); segend = ends_yin(max_idx_yin);
        yin_segprocessed(segstart+1:segend) = track_yin_masked(segstart+1:segend);
    end
end
yin_segprocessed(yin_segprocessed == 0) = NaN;
n_yin = (yin_indices - 1) / fd;

% ========================================================================
%% 4b. CREPE Track Processing (Deep Learning Network)
% ========================================================================
[f0_crepe, loc_crepe] = pitchnn(x, fs, 'ConfidenceThreshold', 0, 'ModelCapacity', 'full');
track_crepe = f0_crepe(:);

% Compensate for the 32 ms causal shift (half of the 64 ms window)
t_crepe = loc_crepe(:) - 0.032; 

% Align shifted neural net timestamps to your global activity mask
crepe_indices = min(max(round(t_crepe * fd) + 1, 1), length(activity_mask));
mask_crepe = activity_mask(crepe_indices);
track_crepe_masked = track_crepe .* mask_crepe(:);

zeromask_crepe = (track_crepe_masked == 0);
ends_crepe = find(diff(zeromask_crepe, 1) == 1);
starts_crepe = find(diff(zeromask_crepe, 1) == -1);
if ~isempty(zeromask_crepe) && zeromask_crepe(1) == 0, starts_crepe = [0; starts_crepe]; end
if length(ends_crepe) < length(starts_crepe), ends_crepe = [ends_crepe; length(track_crepe_masked)]; end
crepe_segprocessed = zeros(length(track_crepe_masked), 1);
if ~isempty(starts_crepe)
    all_seglens_crepe = ends_crepe - starts_crepe;
    [max_len_crepe, max_idx_crepe] = max(all_seglens_crepe);
    if max_len_crepe > 2
        segstart = starts_crepe(max_idx_crepe); segend = ends_crepe(max_idx_crepe);
        crepe_segprocessed(segstart+1:segend) = track_crepe_masked(segstart+1:segend);
    end
end
crepe_segprocessed(crepe_segprocessed == 0) = NaN;

% ========================================================================
%% Displayed Graphic (Visual Layout Maintained)
% ========================================================================
figure(Position=[100, 100, 1100, 800])
[s1, f1, t1] = stft(x, fs, "Window", hamming(round(0.2*fs)), "OverlapLength", round(0.2*fs/2), "FrequencyRange", "onesided");
flm = [find(f1>100, 1), find(f1>400, 1)];
imagesc(t1, f1(flm(1):flm(2)), 20*log10(abs(s1(flm(1):flm(2), :))))
axis xy;
colormap("jet");
hold on

% Plot all dynamic frequency tracking vectors cleanly over the canvas
plot(n_yin, yin_segprocessed, Color="B", LineWidth=1.5, DisplayName="YIN Estimate")
plot(n_acorr, acorr_segprocessed, Color="R", LineWidth=1.5, DisplayName="Autocorrelation Estimate")
plot(t_stft, stft_segprocessed, Color="M", LineWidth=1.5, DisplayName="STFT Estimate")
plot(n, leemorf_segfiltered, Color="K", LineWidth=1.5, DisplayName="Lee-Morf Estimate")
plot(t_crepe, crepe_segprocessed, Color=[0.93, 0.69, 0.13], LineWidth=1.5, DisplayName="CREPE Estimate")

legend('Location', 'northwest')
cb = colorbar(gca, "eastoutside");
cb.Label.String = 'Spectrogram magnitude [dB]';
xlabel("Recording time [s]")
ylabel("Frequency [Hz]")
exportgraphics(gcf, sprintf("IMG_OUT/ESTIMATORS_DATASET_%s.pdf", file))

% ========================================================================
%% Reference Script Helper Functions
% ========================================================================
function [estimate, indices] = f0_estimates(signal, fs, fmin, fmax, winlen)
    estidx = 1;
    for i = 1:fs*winlen:length(signal)-fs*winlen
        signal_window = signal(i:i+fs*winlen);
        estimate(estidx) = estimate_f0_autocorr(signal_window, fs, fmin, fmax);
        indices(estidx) = i;
        estidx = estidx + 1;
    end
end
%%
function f0 = estimate_f0_autocorr(x, fs, fmin, fmax)
    x = x(:) - mean(x);
    
    [r, lags] = xcorr(x, 'biased');
    
    r = r(lags >= 0);
    
    min_lag = max(ceil(fs / fmax), 2);
    max_lag = min(floor(fs / fmin), length(r) - 2); 
    
    search_indices = (min_lag + 1):(max_lag + 1);
    
    [~, max_idx] = max(r(search_indices));
    
    best_idx = search_indices(max_idx);
    
    y1 = r(best_idx - 1);
    y2 = r(best_idx);
    y3 = r(best_idx + 1);
    
    p = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3);
    
    exact_lag = (best_idx - 1) + p;
    
    f0 = fs / exact_lag;
end