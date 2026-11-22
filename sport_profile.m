%% sport_profile.m — synthetic 1800 s Sport (aggressive) vehicle-speed profile
% Based on the "Hill" reference condition, but with gradient = 0 in the model,
% Sport is represented as aggressive driving: hard accelerations, high speeds.

t   = (0:1800)';
aUp = 1.8;   % m/s^2 acceleration (model max = 2 m/s^2)
aDn = 2.0;   % m/s^2 braking

% [target speed (m/s), hold time (s)] — cycled until ~1800 s
seg = [25 35;  0 12; 33 50; 18 25; 35 70;  0 15; 28 40; 34 60; 15 25;  0 12; ...
       30 45; 35 80; 22 30;  0 15; 32 55; 26 35; 35 75; 10 20];

T = 0; V = 0; tt = 0; vv = 0; k = 0;
while true
    k  = k + 1;
    s  = seg(mod(k-1, size(seg,1)) + 1, :);
    vt = s(1);  hd = s(2);
    a  = aUp*(vt > vv) + aDn*(vt <= vv);
    tr = abs(vt - vv) / a;                       % ramp time
    if tt + tr + hd + vt/aDn + 5 > 1800, break; end
    T(end+1) = tt + tr;       V(end+1) = vt;     %#ok<SAGROW>
    T(end+1) = tt + tr + hd;  V(end+1) = vt;     %#ok<SAGROW>
    tt = tt + tr + hd;  vv = vt;
end
T(end+1) = tt + vv/aDn;  V(end+1) = 0;           % final stop
T(end+1) = 1800;         V(end+1) = 0;

v = interp1(T, V, t, 'linear');
v = movmean(v, 3);                               % soften acceleration steps
v = max(v, 0);  v(1) = 0;  v(end) = 0;

SportVehicleSpeed_TS      = timeseries(v, t);
SportVehicleSpeed_TS.Name = 'VehicleSpeed';

fprintf('Max speed  : %.1f m/s (%.0f km/h)\n', max(v), max(v)*3.6);
fprintf('Max accel  : %.2f m/s^2\n', max(diff(v)));
fprintf('Max decel  : %.2f m/s^2\n', min(diff(v)));
fprintf('End speed  : %.2f m/s at %d s\n', v(end), t(end));

figure; plot(t, v, 'LineWidth', 1.2); grid on
xlabel('Time (s)'); ylabel('Vehicle Speed (m/s)');
title('Sport Drive Cycle (synthetic, 1800 s)');

%% Next: import SportVehicleSpeed_TS into a new "SportProfile" scenario in
% Signal Editor (same way as NormalProfile), so it shows SportProfile > VehicleSpeed.
%
% Option B (if GUI import is fiddly): add the scenario straight into the
% Signal Editor's MAT file, then close/reopen the block to refresh.
% seBlk  = 'ESSSimulation/Signal Editor';      % adjust to your block's name
% seFile = get_param(seBlk, 'FileName');
% SportProfile = Simulink.SimulationData.Dataset;
% SportProfile = SportProfile.addElement(SportVehicleSpeed_TS, 'VehicleSpeed');
% save(seFile, 'SportProfile', '-append');
