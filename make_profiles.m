%% make_profiles.m — builds EV_DriveProfiles.mat for the Signal Editor
mdl     = 'ESSSimulation';
seBlk   = [mdl '/Signal Editor'];
matFile = fullfile(pwd, 'EV_DriveProfiles.mat');
t = (0:1800)';

%% NORMAL (Highway) — profile from the handover
tKey = [0 100 220 350 500 650 800 950 1100 1250 1400 1550 1700 1800];
vKey = [0  18  24  28  30  26  32  30   34   31   35   32   28    0];
vN = max(interp1(tKey, vKey, t, 'pchip'), 0);
NormalProfile = makeScenario(vN, t);

%% SPORT (Hill-reference, aggressive driving)
segS = [25 35;  0 12; 33 50; 18 25; 35 70;  0 15; 28 40; 34 60; 15 25;  0 12; ...
        30 45; 35 80; 22 30;  0 15; 32 55; 26 35; 35 75; 10 20];
vS = buildCycle(segS, 1.8, 2.0, t);
SportProfile = makeScenario(vS, t);

%% ECO (Traffic-reference: low speed, stop-and-go)
segE = [ 8 40; 0 30; 12 60; 6 30; 0 25; 14 90; 10 40; 0 35; 11 70; 0 30; ...
        13 80; 7 35; 0 30; 9 50; 14 70; 0 40];
vE = buildCycle(segE, 0.9, 1.2, t);
EcoProfile = makeScenario(vE, t);

%% Save
save(matFile, 'EcoProfile', 'NormalProfile', 'SportProfile');
fprintf('Created %s with Eco, Normal, Sport.\n', matFile);

%% Point the Signal Editor at the new file
load_system(mdl);
set_param(seBlk, 'FileName', matFile);
set_param(seBlk, 'ActiveScenario', 'EcoProfile');
set_param(mdl, 'StopTime', '1800');
fprintf('Signal Editor -> EcoProfile, stop time 1800 s\n');

%% Check plot
S = load(matFile);
figure('Name','Drive cycles'); hold on; grid on
names = fieldnames(S);
for k = 1:numel(names)
    ts = S.(names{k}).get('VehicleSpeed').Values;
    plot(ts.Time, ts.Data*3.6, 'DisplayName', names{k});
    fprintf('%-14s max %5.1f km/h, end speed %.2f m/s\n', names{k}, max(ts.Data)*3.6, ts.Data(end));
end
xlabel('Time (s)'); ylabel('Speed (km/h)'); legend; title('Drive cycles (1800 s)');

%% ---------- local functions ----------
function ds = makeScenario(v, t)
    ts = timeseries(v, t);
    ts.Name = 'VehicleSpeed';
    ts.DataInfo.Units = 'm/s';
    ds = Simulink.SimulationData.Dataset;
    ds = ds.addElement(ts, 'VehicleSpeed');
end

function v = buildCycle(seg, aUp, aDn, t)
    T = 0; V = 0; tt = 0; vv = 0; k = 0;
    while true
        k  = k + 1;
        s  = seg(mod(k-1, size(seg,1)) + 1, :);
        vt = s(1);  hd = s(2);
        a  = aUp*(vt > vv) + aDn*(vt <= vv);
        tr = abs(vt - vv) / a;
        if tt + tr + hd + vt/aDn + 5 > t(end), break; end
        T(end+1) = tt + tr;       V(end+1) = vt;   %#ok<AGROW>
        T(end+1) = tt + tr + hd;  V(end+1) = vt;   %#ok<AGROW>
        tt = tt + tr + hd;  vv = vt;
    end
    if vv > 0
        T(end+1) = tt + vv/aDn;  V(end+1) = 0;
    end
    if T(end) < t(end)
        T(end+1) = t(end);  V(end+1) = 0;
    end
    [T, iu] = unique(T, 'stable');  V = V(iu);
    v = interp1(T, V, t, 'linear');
    v = max(movmean(v, 3), 0);  v(1) = 0;  v(end) = 0;
end