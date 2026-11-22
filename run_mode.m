%% run_mode.m — runs ESSSimulation for one drive mode and builds the comparison table
mode  = 'Sport';                        % 'Eco' | 'Normal' | 'Sport'
mdl   = 'ESSSimulation';
seBlk = [mdl '/Signal Editor'];
load_system(mdl);

% block output port -> logged signal name
logMap = {'Signal Editor',          'VehicleSpeed';
          'Discrete Derivative',    'Acceleration';
          'Sum',                    'TractiveForce';
          'Mechanical Power (W)',   'MechPower';
          'Saturation',             'BattPower';
          'PS-Simulink Converter1', 'Current';
          'PS-Simulink Converter',  'Voltage';
          'P45B_98S30P',            'SOC';
          'Gain',                   'Energy';        % kWh
          'Ah Calculation',         'Ah'};
for k = 1:size(logMap,1)
    ph = get_param([mdl '/' logMap{k,1}], 'PortHandles');
    set_param(ph.Outport(1), 'DataLogging', 'on', ...
        'DataLoggingNameMode', 'Custom', 'DataLoggingName', logMap{k,2});
end

set_param(seBlk, 'ActiveScenario', [mode 'Profile']);
out = sim(mdl, 'StopTime', '1800');
L   = out.logsout;
sig = @(n) squeeze(L.getElement(n).Values.Data);
tm  = L.getElement('VehicleSpeed').Values.Time;

v  = sig('VehicleSpeed');   a  = sig('Acceleration');
F  = sig('TractiveForce');  Pm = sig('MechPower');
Pb = sig('BattPower');      I  = sig('Current');
V  = sig('Voltage');        S  = sig('SOC');       E = sig('Energy');
Ah = sig('Ah');
if ~isvector(S)                          % one SOC per module -> average
    if size(S,1) == numel(tm), S = mean(S,2); else, S = mean(S,1)'; end
end
if max(S) <= 1.01, S = 100*S; end        % 0–1 SOC -> %

r.Mode              = mode;
r.MaxSpeed_kmh      = max(v)*3.6;
r.PeakAccel_ms2     = max(a);
r.PeakTractiveF_N   = max(F);
r.PeakMechPower_kW  = max(Pm)/1e3;
r.PeakBattPower_kW  = max(Pb)/1e3;
r.PeakCurrent_A     = max(I);
r.PeakCellCurrent_A = max(I)/30;
r.Vinit_V           = V(1);
r.Vfinal_V          = V(end);
r.Vmin_V            = min(V);
r.SOCinit_pct       = S(1);
r.SOCfinal_pct      = S(end);
r.SOCused_pct       = S(1) - S(end);
r.Energy_kWh        = E(end) - E(1);
r.Charge_Ah         = Ah(end) - Ah(1);

% --- save into cumulative results ---
f = 'mode_results.mat';
if isfile(f), load(f, 'R'); end
R.(mode) = r;
save(f, 'R');
c = struct2cell(R);
T = struct2table([c{:}]);
disp(T)
writetable(T, 'mode_comparison.xlsx');

% --- plots for the report ---
fig = figure('Name', mode, 'Position', [100 100 1200 800]);
names = {'Vehicle speed (m/s)','Acceleration (m/s^2)','Tractive force (N)', ...
         'Mech power (kW)','Battery power (kW)','Current (A)', ...
         'Voltage (V)','SOC (%)','Energy (kWh)'};
data  = {v, a, F, Pm/1e3, Pb/1e3, I, V, S, E};
for k = 1:9
    subplot(3,3,k); plot(tm, data{k}); grid on
    title(names{k}); xlabel('Time (s)');
end
sgtitle([mode ' mode — 1800 s']);
exportgraphics(fig, [mode '_results.png'], 'Resolution', 200);