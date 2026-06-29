clearvars;
hop_len = 10;
max_order_to_test = 50;
fd = 2000; %[Hz]
winlens = 10:100; %window lengths in milliseconds
set(0,'DefaultFigureVisible','off')
max_win_samples = floor(max(winlens) * fd / 1000);
NFFT = 2^nextpow2(max_win_samples);

window_test_length = 50;%[ms]

scenarios = ["variable buzz worker", "stable buzz worker", "variable buzz drone", "stable buzz drone"]; 

for scenario = scenarios
    signal_testfolder = sprintf("Sound_verif_set/%s", scenario);

    specimen_dir = dir(signal_testfolder);
    dirFlags = [specimen_dir.isdir];
    subDirs = specimen_dir(dirFlags);

    specimens = string({subDirs(3:end).name});

    for specimen=specimens
        original_files = dir(sprintf("%s/%s/*.wav", signal_testfolder, specimen));

        for k=1:length(original_files)

            filename = original_files(k).name;
            sanitized_filename = strrep(strrep(filename, "_", " "),".wav", "")
            signal_path=sprintf("%s/%s/%s", signal_testfolder, specimen, filename);

            DATA_PATH = sprintf("DATA_OUT_AIC/%s/%s", scenario, specimen);

            mkdir(DATA_PATH);

            %read and filter signal
            [x,fs]=audioread(signal_path);
            x = rescale(x);
            [B,A] = ellip(3,0.1,50,2*100/fs,'high');
            x = filtfilt (B,A,x); clear B A
            y = resample(x,fd,fs);

            if not((length(y) - 4*max(winlens*fd/1000))<fd)
                activity_mask = zeros();
                winidx = 1;
                for winstart = 1:hop_len:(length(y)-2*max(winlens*fd/1000)-1)
                    rms_win = rms(y(winstart:winstart+10));
                    activity_mask(winidx) = rms_win;
                    winidx = winidx+1;
                end

                aic_res = [];
                optimal_AIC = [];

                winlen = window_test_length;
                win_samples = floor(winlen*fd/1000);
                fprintf("Now at window length: %i\n", winlen);

                window_indices = 1:hop_len:(length(y)-4*max(winlens*fd/1000)-1);

                parfor window_idx = 1:(length(window_indices)-1)

                    window_function = hann(win_samples+1);
                    windowed = y(window_indices(window_idx):window_indices(window_idx)+win_samples).*window_function;

                    bounds_AR_AIC = 1:min(length(windowed),max_order_to_test);

                    aic_res = [];
                    for p = bounds_AR_AIC
                        [coeffs, variance] = aryule(windowed, p);
                        aic_res(p) = 2*(p+1) + length(windowed)*log(variance);
                    end

                    [M, I] = min(aic_res);
                    optimal_AIC(window_idx) = bounds_AR_AIC(I);
                end
                save(sprintf("%s/RESULTS_AIC_%s_%s.mat", DATA_PATH, scenario, sanitized_filename), ...
                    'window_indices', "optimal_AIC", "activity_mask");
            end
        end
    end
end
