close all; clear all; clc;
set(groot,'defaultAxesFontSize', 14)
set(0, 'DefaultAxesFontName', 'Times');

%% Setup & Universal Configurations
rng(0, 'twister'); % Set random seed for repeatability
fd = 2000;
fs = 12000;
SNR = 12; % Fixed at 12 dB for the sensitivity study

% Sensitivity Study Grid
tau_test_ms = 20:5:80;      % 20 to 80 ms, step of 1
order_test = 10:5:30;       % Order 10 to 30, step of 1

% Initialize output variables
rmse_heatmap = zeros(length(order_test), length(tau_test_ms));
results_history = struct('Tau_ms', {}, 'Model_Order', {}, 'RMSE', {});

%% 1. Generate 12 dB Signal (Calculated ONCE)
fprintf('Generating 12 dB SNR signal...\n');
A = sqrt(2*((0.5^2)/3)*10^(SNR/10));
t_gen = 0:1/fs:10;
f_inst = 200 + 50 * sawtooth(2 * pi * 1 * t_gen, 0.5); 
t_ref = t_gen; f_ref = f_inst; 

sin_component = A*sin(cumsum(2 * pi * f_inst / fs));
noise = rand(1, length(t_gen)) - 0.5;

x = (sin_component + noise)' .* tukeywin(length(t_gen));

% Rescale and highpass filter signal
x = rescale(x);
[B, A_filt] = ellip(3, 0.1, 50, 2*100/fs, 'high');
x = filtfilt(B, A_filt, x);
y = resample(x, fd, fs);

% Setup time arrays
T = size(x,1); t = (0:T-1)'/fs; 
N = size(y,1); n = (0:N-1)'/fd;

% Calculate Activity mask
activity_mask = zeros(length(y), 1);
for winstart = 1:100:(length(y)-100)
    rms_win = rms(y(winstart:winstart+100));
    activity_mask(winstart:winstart+100) = rms_win > 0.01;
end

% Pre-compute low-pass filter for the track
[b2, a2] = ellip(5, 1, 80, 0.1, "low");

%% 2. Run Sensitivity Grid Search
fprintf('Starting Lee-Morf parameter grid search (%i iterations)...\n', length(tau_test_ms) * length(order_test));

loop_idx = 1;
total_iters = length(tau_test_ms) * length(order_test);

for idx_tau = 1:length(tau_test_ms)
    tau = tau_test_ms(idx_tau) / 1000; % Convert to seconds
    
    for idx_order = 1:length(order_test)
        AR_P = order_test(idx_order);
        lambda = 1 - (1/fd)/((1/fd)+tau);
        
        % Run LEEMORF
        [Ro, ~, ~] = lee_morf_new(y, AR_P, lambda);
        A_coef = ktoa(Ro');
        SD = fft(A_coef, 2048);
        S = 20*log10(1./abs(SD(1:1024, :)));
        f = fd*(0:1023)/2048;
        M = S > prctile(S, 95, "all");
        SF = S; SF(~M) = 0;
        rdg_leemorf = tfridge(SF(1:300, :), f(1:300), 0);
        rdg_leemorf = rdg_leemorf(:);
        
        % Process Mask and Segment
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
        if ~isempty(segment_starts)
            all_seglens = segment_ends - segment_starts;
            [max_len, max_idx] = max(all_seglens);
            if max_len > 2
                segstart = segment_starts(max_idx); segend = segment_ends(max_idx);
                leemorf_segfiltered(segstart+1:segend) = filtfilt(b2, a2, rdg_leemorf(segstart+1:segend));
            end
        end
        leemorf_segfiltered(leemorf_segfiltered == 0) = NaN;
        
        % Shift LEEMORF to offset causal group delay
        leemorf_shift_samples = round(0.025 * fd);
        if length(leemorf_segfiltered) > leemorf_shift_samples
            leemorf_segfiltered = [leemorf_segfiltered(leemorf_shift_samples+1:end); NaN(leemorf_shift_samples, 1)];
        end
        
        % Calculate standard RMSE (omitted bootstrap to save time during grid search)
        valid_leemorf = ~isnan(leemorf_segfiltered);
        t_leemorf_valid = n(valid_leemorf);
        f_true_leemorf = interp1(t_ref(:), f_ref(:), t_leemorf_valid(:), 'linear');
        f_est_leemorf = leemorf_segfiltered(valid_leemorf);
        
        errors = f_true_leemorf(:) - f_est_leemorf(:);
        rmse_val = sqrt(mean(errors.^2));
        
        % Store Results
        rmse_heatmap(idx_order, idx_tau) = rmse_val;
        
        results_history(loop_idx).Tau_ms = tau_test_ms(idx_tau);
        results_history(loop_idx).Model_Order = AR_P;
        results_history(loop_idx).RMSE = rmse_val;
        
        % Progress counter
        if mod(loop_idx, 100) == 0
            fprintf('Processed %i / %i configurations...\n', loop_idx, total_iters);
        end
        loop_idx = loop_idx + 1;
    end
end
fprintf('Grid search complete.\n');

%% 3. Generate 2D Heatmap with Overlay Numbers
figure('Position', [100, 100, 850, 650]);
imagesc(tau_test_ms, order_test, rmse_heatmap);
axis xy; 
colormap('jet'); 
cb = colorbar;
cb.Label.String = 'Tracking RMSE [Hz]';
xlabel('Effective Time Window \tau [ms]');
ylabel('Model Order');

min_val = min(rmse_heatmap(:), [], 'omitnan');
max_val = max(rmse_heatmap(:), [], 'omitnan');
mid_val = min_val + (max_val - min_val) / 2;

for idx_tau = 1:length(tau_test_ms)
    for idx_order = 1:length(order_test)
        val = rmse_heatmap(idx_order, idx_tau);
        
        if isnan(val), continue; end 
        
        if val > mid_val
            txt_color = 'black'; 
        else
            txt_color = 'white';
        end
        
        text(tau_test_ms(idx_tau), order_test(idx_order), sprintf('%.2f', val), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', txt_color, ...
            'FontWeight', 'bold', ...
            'FontSize', 10);
    end
end
% Save the figure
if ~exist('IMG_OUT', 'dir'), mkdir('IMG_OUT'); end
exportgraphics(gcf, "IMG_OUT/LEEMORF_Sensitivity_Heatmap.pdf");

%% 4. Export to Excel
output_excel_filename = 'LeeMorf_Sensitivity_Results.xlsx';
results_table = struct2table(results_history);
writetable(results_table, output_excel_filename, 'Sheet', 'Sensitivity Grid');
fprintf('\n>>> Successfully saved sensitivity metrics to: %s\n', output_excel_filename);