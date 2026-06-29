clearvars;
hop_len = 10;
max_order_to_test = 30;
fd = 12000; %[Hz]
winlens = 10:100; %window lengths in milliseconds
chosen_time_for_AR_order = find(winlens==50);
set(0,'DefaultFigureVisible','on')
set(0, 'DefaultAxesFontName', 'Times');

scenarios = ["variable buzz worker","stable buzz worker", "variable buzz drone", "stable buzz drone"];
scenarios_names = ["Variable buzz worker","Stable buzz worker", "Variable buzz drone", "Stable buzz drone"];

for scenario_idx = 1:length(scenarios)

    scenario = scenarios(scenario_idx);
    signal_testfolder = sprintf("Sound_verif_set/%s", scenario);
    specimen_dir = dir(signal_testfolder);
    dirFlags = [specimen_dir.isdir];
    subDirs = specimen_dir(dirFlags);

    specimens = string({subDirs(3:end).name});
    env_win_predictor = [];
    optimal_aics = [];
    histo_index = 1;
    winlen_edges = 1:max(winlens);

    for specimen_idx=1:length(specimens)
        specimen=specimens(specimen_idx);
        original_files = dir(sprintf("%s/%s/*.wav", signal_testfolder, specimen));
        percentile_stationary_window = [];
        for k=1:length(original_files)
            filename = original_files(k).name;
            sanitized_filename = strrep(strrep(filename, "_", " "),".wav", "")
            signal_path=sprintf("%s/%s/%s", signal_testfolder, specimen, filename);

            DATA_PATH = sprintf("DATA_OUT/%s/%s", scenario, specimen);
            AIC_PATH = sprintf("DATA_OUT_AIC/%s/%s", scenario, specimen);

            %load the results
            %check if file exists
            if isfile(sprintf("%s/RESULTS_%s_%s.mat", DATA_PATH, scenario, sanitized_filename)) && isfile(sprintf("%s/RESULTS_AIC_%s_%s.mat", AIC_PATH, scenario, sanitized_filename))

                load(sprintf("%s/RESULTS_%s_%s.mat", DATA_PATH, scenario, sanitized_filename), ...
                    'window_indices', "activity_mask", "renentr_matrix");

                load(sprintf("%s/RESULTS_AIC_%s_%s.mat", AIC_PATH, scenario, sanitized_filename), ...
                    "optimal_AIC");

                for col = 1:width(renentr_matrix)
                    renemat = renentr_matrix(:, col);
                    [pk, ~] = min(renemat);
                    loc = find(renemat<pk+0.05*pk, 1); % shortest window within 5% of minimum entropy.

                    if ~isempty(loc)
                        env_win(col) = winlens(loc);
                    else
                        env_win(col) = 0;
                    end
                end
                env_win_hist(histo_index, :) = histcounts(env_win, 1:1:max(winlens));
                env_win_predictor = [env_win_predictor, env_win];

                optimal_aics = [optimal_aics, optimal_AIC];
                optimal_aic_hist(histo_index, :) = histcounts(optimal_AIC, 1:1:50);
                % pause(10);
                histo_index = histo_index+1;
            end
        end
    end

    env_win_all(scenario_idx, :) = {env_win_predictor};
    optimal_aics_all(scenario_idx, :) = {optimal_aics};
    env_win_histo_all(scenario_idx, :) = sum(env_win_hist);
    optimal_aics_histo_all(scenario_idx, :) = sum(optimal_aic_hist);
end

figure(Position=[100, 100, 1100, 800])
for scenario_idx = 1:length(scenarios)

    medianval = median(cell2mat(env_win_all(scenario_idx)), "omitmissing");
    prc_25 = prctile(cell2mat(env_win_all(scenario_idx)), 25);
    prc_75 = prctile(cell2mat(env_win_all(scenario_idx)), 75);

    subplot(2, 2, scenario_idx)
    bar(winlen_edges(1:end-1), env_win_histo_all(scenario_idx, :)/sum(env_win_histo_all(scenario_idx, :)), "LineStyle","none")

    xline(prc_25, "-.", "Color","r", "LineWidth",2)
    xline(medianval, "--", "Color","g", "LineWidth",2)
    xline(prc_75, "-.", "Color","r", "LineWidth",2)

    title(scenarios_names(scenario_idx))
    xlabel("Window length [ms]")
    ylabel("Distribution value")
    xlim([10, 70])
    ylim([0, 0.15])
    grid on
    legend(["Distribution", sprintf("P_{25}=%.1f", prc_25), sprintf("Median=%.1f", medianval), sprintf("P_{75}=%.1f", prc_75)], "Location","northwest")

end

print(gcf, "IMG_OUT/Distribution_results", "-dpng", "-r600")
exportgraphics(gcf, "IMG_OUT/Distribution_results.pdf")

figure(Position=[100, 100, 1100, 800])
for scenario_idx = 1:length(scenarios)

    medianval = median(cell2mat(optimal_aics_all(scenario_idx)), "omitmissing");
    prc_5 = prctile(cell2mat(optimal_aics_all(scenario_idx)), 5);
    prc_95 = prctile(cell2mat(optimal_aics_all(scenario_idx)), 95);
    
    subplot(2, 2, scenario_idx)
    bar(1:49, optimal_aics_histo_all(scenario_idx, :)/sum(optimal_aics_histo_all(scenario_idx, :)), "LineStyle","none")

    xline(prc_5, "-.", "Color","r", "LineWidth",2)
    xline(medianval, "--", "Color","g", "LineWidth",2)
    xline(prc_95, "-.", "Color","r", "LineWidth",2)

    title(scenarios_names(scenario_idx))
    xlabel("Optimal AR model order")
    ylabel("Distribution value")
    xlim([1, 50])
    ylim([0, 0.15])
    grid on
    legend(["Distribution", sprintf("P_{5}=%.1f", prc_5), sprintf("Median=%.1f", medianval), sprintf("P_{95}=%.1f", prc_95)], "Location","northeast")
end

print(gcf, "IMG_OUT/Distribution_results_AIC", "-dpng", "-r600")
exportgraphics(gcf, "IMG_OUT/Distribution_results_AIC.pdf")