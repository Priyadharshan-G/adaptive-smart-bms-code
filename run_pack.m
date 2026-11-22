function T = run_pack(packTag)
%RUN_PACK Run Eco, Normal and Sport on one pack model and export everything.
%
%   run_pack('NCR')    % ESSSimulation_NCR  + NCR18650BD_98S45P (45P)
%   run_pack('LFP')    % ESSSimulation_LFP  + ANR26650_120S52P  (52P)
%   run_pack('P45B')   % original model, re-run for comparison (30P)
%
% Writes (all with the pack name in front, so the P45B files are never overwritten):
%   <tag>_Eco_timeseries.csv, <tag>_Normal_timeseries.csv, <tag>_Sport_timeseries.csv
%       same columns as the original CSVs, so the Python BMS can read them
%   <tag>_Eco_results.png ...           9-panel figure per mode
%   <tag>_mode_comparison.xlsx          one row per mode
% Run from the project folder (D:\Electrical).

    cfg = struct( ...
        'P45B', struct('mdl','ESSSimulation',     'pack','P45B_98S30P',       'Np',30), ...
        'NCR',  struct('mdl','ESSSimulation_NCR', 'pack','NCR18650BD_98S45P', 'Np',45), ...
        'LFP',  struct('mdl','ESSSimulation_LFP', 'pack','ANR26650_108S57P',  'Np',57));
    if ~isfield(cfg, packTag)
        error('packTag must be ''P45B'', ''NCR'' or ''LFP''.');
    end
    c     = cfg.(packTag);
    mdl   = c.mdl;
    pack  = [mdl '/' c.pack];
    seBlk = [mdl '/Signal Editor'];
    load_system(mdl);

    if getSimulinkBlockHandle(pack) == -1
        error('Block %s not found. Rename the pack block in the model to %s.', pack, c.pack);
    end

    % --- logging on the same blocks as run_mode.m ---
    logMap = {'Signal Editor',          'VehicleSpeed';
              'Discrete Derivative',    'Acceleration';
              'Sum',                    'TractiveForce';
              'Mechanical Power (W)',   'MechPower';
              'Saturation',             'BattPower';
              'PS-Simulink Converter1', 'Current';
              'PS-Simulink Converter',  'Voltage';
              'Gain',                   'Energy';        % kWh
              'Ah Calculation',         'Ah'};
    for k = 1:size(logMap,1)
        ph = get_param([mdl '/' logMap{k,1}], 'PortHandles');
        set_param(ph.Outport(1), 'DataLogging', 'on', ...
            'DataLoggingNameMode', 'Custom', 'DataLoggingName', logMap{k,2});
    end

    % --- SOC: P45B pack has a 'socOutCell' (SOC) port; Battery Builder packs
    %     have 'socCell' (one value per module). Use whichever exists. ---
    socPort = [];
    op = find_system(pack, 'SearchDepth', 1, 'BlockType', 'Outport');
    names = cellfun(@(b) get_param(b, 'Name'), op, 'UniformOutput', false);
    for want = {'socOutCell', 'socCell'}
        k = find(strcmpi(names, want{1}), 1);
        if ~isempty(k)
            socPort = str2double(get_param(op{k}, 'Port'));
            fprintf('SOC taken from pack port "%s" (port %d)\n', names{k}, socPort);
            break
        end
    end
    if isempty(socPort)
        error('Pack %s has no socOutCell or socCell output.', pack);
    end
    ph = get_param(pack, 'PortHandles');
    for k = 1:numel(ph.Outport)                          % no other pack port may log
        set_param(ph.Outport(k), 'DataLogging', 'off');
    end
    set_param(ph.Outport(socPort), 'DataLogging', 'on', ...
        'DataLoggingNameMode', 'Custom', 'DataLoggingName', 'PackSOC');

    % --- check the drive-profile file before spending minutes simulating ---
    f = get_param(seBlk, 'FileName');
    fprintf('Signal Editor file: %s  ->  %s\n', f, which(f));
    w = whos('-file', which(f));
    missing = setdiff({'EcoProfile','NormalProfile','SportProfile'}, {w.name});
    if ~isempty(missing)
        error(['%s has no %s. Copy EV_DriveProfiles.mat from the repo into D:\\Electrical ' ...
               '(overwrite) and check  which -all EV_DriveProfiles.mat'], f, strjoin(missing, ', '));
    end

    modes = {'Eco','Normal','Sport'};
    rows  = cell(1, 3);
    for m = 1:3
        mode = modes{m};
        fprintf('\n[%s] Running %s ...\n', packTag, mode);
        set_param(seBlk, 'ActiveScenario', [mode 'Profile']);
        out = sim(mdl, 'StopTime', '1800');
        L   = out.logsout;
        % put every signal on one 10 s grid (Sport can log extra solver steps)
        tm  = (0:10:1800)';
        sig = @(n) resampleTo(L.getElement(n).Values, tm);

        v  = sig('VehicleSpeed');   a  = sig('Acceleration');
        F  = sig('TractiveForce');  Pm = sig('MechPower');
        Pb = sig('BattPower');      I  = sig('Current');
        V  = sig('Voltage');        S  = sig('PackSOC');   E = sig('Energy');
        Ah = sig('Ah');
        if ~isvector(S)                                    % one SOC per module -> average
            if size(S,1) == numel(tm), S = mean(S,2); else, S = mean(S,1)'; end
        end
        if max(S) <= 1.01, S = 100*S; end                  % 0-1 -> %
        if min(S) < 0 || max(S) > 101
            error('Logged SOC runs from %.2f to %.2f %% - wrong pack port..', min(S), max(S));
        end
        if strcmp(mode, 'Eco') && max(v)*3.6 > 60
            warning('Eco top speed is %.0f km/h (expected 56). The Signal Editor file is the old one.', max(v)*3.6);
        end

        % --- CSV in the same format as the original *_timeseries.csv ---
        Temp = 25*ones(size(tm));                          % isothermal, as before
        tbl = table(tm, V(:), I(:), S(:), Temp, v(:), 'VariableNames', ...
            {'Time_s','Voltage_V','Current_A','SOC_pct','Temp_degC','Speed_ms'});
        writetable(tbl, sprintf('%s_%s_timeseries.csv', packTag, mode));

        % --- summary row ---
        r = struct();
        r.Pack              = string(packTag);
        r.Mode              = string(mode);
        r.PeakBattPower_kW  = max(Pb)/1e3;
        r.PeakCurrent_A     = max(I);
        r.PeakCellCurrent_A = max(I)/c.Np;
        r.Vinit_V           = V(1);
        r.Vfinal_V          = V(end);
        r.Vmin_V            = min(V);
        r.SOCinit_pct       = S(1);
        r.SOCfinal_pct      = S(end);
        r.SOCused_pct       = S(1) - S(end);
        r.Energy_kWh        = E(end) - E(1);
        r.Charge_Ah         = Ah(end) - Ah(1);
        rows{m} = r;
        fprintf('  Vinit %.1f V | Vmin %.1f V | SOC %.2f -> %.2f %% | %.2f kWh\n', ...
            r.Vinit_V, r.Vmin_V, r.SOCinit_pct, r.SOCfinal_pct, r.Energy_kWh);

        % --- 9-panel figure ---
        fig = figure('Name', [packTag ' ' mode], 'Position', [100 100 1200 800]);
        names = {'Vehicle speed (m/s)','Acceleration (m/s^2)','Tractive force (N)', ...
                 'Mech power (kW)','Battery power (kW)','Current (A)', ...
                 'Voltage (V)','SOC (%)','Energy (kWh)'};
        data  = {v, a, F, Pm/1e3, Pb/1e3, I, V, S, E};
        for k = 1:9
            subplot(3,3,k); plot(tm, data{k}); grid on
            title(names{k}); xlabel('Time (s)');
        end
        sgtitle(sprintf('%s pack - %s mode - 1800 s', packTag, mode));
        exportgraphics(fig, sprintf('%s_%s_results.png', packTag, mode), 'Resolution', 200);
    end

    T = struct2table([rows{:}]);
    disp(T)
    writetable(T, sprintf('%s_mode_comparison.xlsx', packTag));
    fprintf('\nSaved %s_*_timeseries.csv, %s_*_results.png and %s_mode_comparison.xlsx\n', ...
        packTag, packTag, packTag);
end

function y = resampleTo(ts, tq)
% Interpolate a logged timeseries onto tq. Multi-column data (e.g. one SOC per
% module) stays multi-column; repeated time stamps are removed first.
    t = ts.Time(:);
    d = squeeze(ts.Data);
    if isvector(d), d = d(:); elseif size(d,1) ~= numel(t), d = d.'; end
    [t, iu] = unique(t, 'last');
    d = d(iu, :);
    y = interp1(t, d, tq, 'linear', 'extrap');
end
