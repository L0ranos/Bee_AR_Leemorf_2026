close all; clear all;
set(groot,'defaultAxesFontSize', 14)
set(0, 'DefaultAxesFontName', 'Times');

%% Load file
% file = "variable";
file = "stable";
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

activity_mask = zeros(length(y), 1);
for winstart = 1:100:(length(y)-100)
    rms_win = rms(y(winstart:winstart+100));
    activity_mask(winstart:winstart+100) = rms_win > 0.01;
end

lambda = 1 - (1/fd)/((1/fd)+tau);
AR_P = filter_order;

[Ro, ~, ~] = lee_morf_new(y, AR_P, lambda);
A_coef = ktoa(Ro');

e_res = zeros(length(y), 1);
for n_idx = (AR_P + 1):length(y)
    e_res(n_idx) = sum(A_coef(:, n_idx) .* y(n_idx : -1 : n_idx - AR_P));
end
e_active = e_res(activity_mask > 0);

max_lag = 50;
[acf_res, lags_res] = xcorr(e_active, max_lag, 'coeff');
idx_pos = lags_res >= 0;
acf_pos = acf_res(idx_pos);
lags_pos = lags_res(idx_pos);
N_eff = length(e_active);
ci_bound = 1.96 / sqrt(N_eff);

dof_adjusted = max_lag - AR_P;

[h, pValue, stat, cValue] = lbqtest(e_active, 'Lags', max_lag, 'DoF', dof_adjusted);

fprintf('\n========================================================\n');
fprintf('         LJUNG-BOX TEST RESULTS (via lbqtest)           \n');
fprintf('========================================================\n');
fprintf('Signal File:         %s\n', file);
fprintf('Effective Samples:   %d\n', N_eff);
fprintf('Max Lag tested (m):  %d\n', max_lag);
fprintf('Degrees of Freedom:  %d (m - AR_P)\n', dof_adjusted);
fprintf('Q-statistic:         %.4f\n', stat);
fprintf('Critical Value:      %.4f\n', cValue);
fprintf('p-value:             %.4e\n', pValue);
fprintf('--------------------------------------------------------\n');
if h == 1
    fprintf('Conclusion:          REJECT H0 (Residuals are NOT white noise)\n');
else
    fprintf('Conclusion:          FAIL TO REJECT H0 (Residuals are white noise)\n');
end
fprintf('========================================================\n\n');

% --- Plotting ---
figure('Position', [200, 200, 800, 400]);
hold on;
fill([lags_pos(1), lags_pos(end), lags_pos(end), lags_pos(1)], ...
     [ci_bound, ci_bound, -ci_bound, -ci_bound], ...
     [0.85 0.85 0.85], 'EdgeColor', 'none', 'DisplayName', '95% Confidence Interval');
stem(lags_pos, acf_pos, 'k', 'filled', 'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', 'Residual ACF');
plot([lags_pos(1), lags_pos(end)], [0 0], 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');

title(sprintf('ACF of residuals - %s', file));
xlabel('Lag');
ylabel('Autocorrelation');
legend('Location', 'northeast');
grid on; axis tight;
ylim([-max(abs(acf_pos))*1.2, max(abs(acf_pos))*1.2]);
hold off;
exportgraphics(gca, sprintf("IMG_OUT/Whiteness_ACF_%s.pdf", file))