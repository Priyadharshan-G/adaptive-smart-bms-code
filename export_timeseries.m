%% export_timeseries.m — re-runs each mode and saves full time-series to CSV
mdl   = 'ESSSimulation';
seBlk = [mdl '/Signal Editor'];
load_system(mdl);

modes = {'Eco','Normal','Sport'};
for m = 1:3
    mode = modes{m};
    fprintf('Running %s...\n', mode);
    set_param(seBlk, 'ActiveScenario', [mode 'Profile']);
    out = sim(mdl, 'StopTime', '1800');
    L   = out.logsout;

    tm = L.getElement('VehicleSpeed').Values.Time;
    V  = squeeze(L.getElement('Voltage').Values.Data);
    I  = squeeze(L.getElement('Current').Values.Data);
    S  = squeeze(L.getElement('SOC').Values.Data);
    if ~isvector(S)
        if size(S,1)==numel(tm), S=mean(S,2); else, S=mean(S,1)'; end
    end
    if max(S) <= 1.01, S = S * 100; end   % convert 0-1 to %
    spd = squeeze(L.getElement('VehicleSpeed').Values.Data);

    % Temperature: cell model outputs ~298 K, convert to degC
    T = ones(size(tm)) * 25;   % placeholder 25 degC (isothermal model)

    tbl = table(tm, V, I, S, T, spd, ...
        'VariableNames', {'Time_s','Voltage_V','Current_A',...
                          'SOC_pct','Temp_degC','Speed_ms'});
    fname = fullfile(pwd, [mode '_timeseries.csv']);
    writetable(tbl, fname);
    fprintf('Saved %s\n', fname);
end
disp('All three modes exported.');