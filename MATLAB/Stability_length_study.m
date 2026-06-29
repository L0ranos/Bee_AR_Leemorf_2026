clearvars;
hop_len = 10;
max_order_to_test = 20;
fd = 2000; %[Hz]
winlens = 10:100; %window lengths in milliseconds
set(0, 'DefaultFigureVisible', 'off')

max_win_samples = floor(max(winlens) * fd / 1000);
NFFT = 2^nextpow2(max_win_samples);
scenarios = ["variable buzz worker", "stable buzz worker", "variable buzz drone", "stable buzz drone"];

for scenario = scenarios
    signal_testfolder = sprintf("Sound_verif_set/%s", scenario);
    specimen_dir = dir(signal_testfolder);
    dirFlags = [specimen_dir.isdir];
    subDirs = specimen_dir(dirFlags);
    specimens = string({subDirs(3:end).name});
    
    for specimen = specimens
        original_files = dir(sprintf("%s/%s/*.wav", signal_testfolder, specimen));
        
        for k = 1:length(original_files)
            filename = original_files(k).name;
            sanitized_filename = strrep(strrep(filename, "_", " "), ".wav", "");
            signal_path = sprintf("%s/%s/%s", signal_testfolder, specimen, filename);
            IMG_PATH = sprintf("IMG_OUT/%s/%s", scenario, specimen);
            DATA_PATH = sprintf("DATA_OUT/%s/%s", scenario, specimen);
            
            mkdir(IMG_PATH);
            mkdir(DATA_PATH);
            
            % read and filter signal
            [x, fs] = audioread(signal_path);
            x = rescale(x);
            [B, A] = ellip(3, 0.1, 50, 2*100/fs, 'high');
            x = filtfilt(B, A, x); 
            clear B A
            
            y = resample(x, fd, fs);
            
            if not((length(y) - 4*max(winlens*fd/1000)) < fd)
                activity_mask = zeros();
                winidx = 1;
                
                for winstart = 1:hop_len:(length(y) - 2*max(winlens*fd/1000) - 1)
                    rms_win = rms(y(winstart:winstart+10));
                    activity_mask(winidx) = rms_win;
                    winidx = winidx + 1;
                end
                
                aic_res = [];
                diff_variance = [];
                optimal_AIC = [];
                diff_means = [];
                diff_acorr = [];
                diff_f0 = [];
                statio_matrix = [];
                result_MW = [];
                renentr_matrix = [];
                
                for winlen_idx = 1:length(winlens)
                    winlen = winlens(winlen_idx);
                    win_samples = 4 * floor(max(winlens) * fd / 1000);
                    fprintf("Now at window length: %i\n", winlen);
                    window_indices = 1:hop_len:(length(y) - 4*max(winlens*fd/1000) - 1);
                    
                    parfor window_idx = 1:(length(window_indices) - 1)
                        window_function = hann(win_samples + 1);
                        windowed = y(window_indices(window_idx):window_indices(window_idx) + win_samples) .* window_function;
                        
                        current_win_samples = floor(winlen * fd / 1000);
                        current_hop = floor(current_win_samples / 2);
                        
                        [S, F, T] = stft(windowed, fd, "Window", hann(current_win_samples), "OverlapLength", current_hop, "FFTLength", max(winlens)*fd/1000);
                        R = renyi_entropy_function(abs(S.^2), 3, current_hop, fd / max(winlens) * fd / 1000);
                        renentr_matrix(winlen_idx, window_idx) = R;
                        
                        bounds_AR_AIC = 1:min(length(windowed), max_order_to_test);
                        local_aic_res = [];
                        for p = bounds_AR_AIC
                            [coeffs, variance] = aryule(windowed, p);
                            local_aic_res(p) = 2*(p + 1) + length(windowed) * log(variance);
                        end
                        [M, I] = min(local_aic_res);
                        optimal_AIC(winlen_idx, window_idx) = bounds_AR_AIC(I);
                    end
                end
                
                env_win = [];
                env_win_lower = [];
                [b, a] = ellip(5, 5, 50, 0.2);
                
                for col = 1:width(renentr_matrix)
                    renemat = renentr_matrix(:, col);
                    [pk, ~] = min(renemat);
                    loc = find(renemat < pk + 0.05*pk, 1); % shortest window within 5% of minimum entropy.
                    if ~isempty(loc)
                        env_win(col) = winlens(loc);
                    else
                        env_win(col) = 0;
                    end
                end
                
                env_win(env_win == 0) = NaN;
                
                % save the results
                save(sprintf("%s/RESULTS_%s_%s.mat", DATA_PATH, scenario, sanitized_filename), ...
                    'window_indices', "diff_variance", "aic_res", "optimal_AIC", "diff_acorr", "activity_mask", "env_win", "renentr_matrix");
                
                figure(Position=[100, 100, 800, 800])
                
                ax1 = subplot(2, 1, 1);
                [s, f, tspe] = stft(y, fd, "FrequencyRange", "onesided", "Window", hann(128, "periodic"), "OverlapLength", 0);
                colorbar
                imagesc(tspe*fd, f, 10*log10(abs(s).^2))
                set(gca, 'YDir', 'normal')
                xlim([min(window_indices), max(window_indices)])
                title(sprintf("Test signal spectrogram - %s, %s", scenario, sanitized_filename))
                xlabel("Sample")
                ylabel("Frequency [Hz]")
                ylim([0, 1000])
                
                ax2 = subplot(2, 1, 2);
                imagesc(window_indices, winlens, renentr_matrix)
                axis xy
                hold on
                plot(window_indices(1:end-1), env_win, "Color", "k", "DisplayName", "Stationarity estimator")
                xlim([min(window_indices), max(window_indices)])
                legend()
                xlabel("Sample")
                ylabel("Window legth")
                title("Rényi entropy estimate")
                
                exportgraphics(gcf, sprintf("%s/PLOT-%s-%s-%s.png", IMG_PATH, scenario, specimen, sanitized_filename))
            end
        end
    end
end

function R = renyi_entropy_function(x, alpha, a, b)
    x = x(:);
    p = x(x > 0);
    p = p / sum(p); % normalize
    R = zeros(size(alpha));
    for k = 1:length(alpha)
        alf = alpha(k);
        if abs(alf - 1) < 1e-8
            R(k) = -sum(p .* log2(p)) + log2(a * b); % Shannon entropy
        else
            R(k) = log2(sum(p.^alf)) / (1 - alf) + log2(a * b);
        end
    end
end