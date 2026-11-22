%% vehicle_analysis.m — vehicle-side results for every drive profile (no battery needed)
m = 985.8;  kDrag = 0.5147;  Frr = 106.3;  eta = 0.9;
Vbat = 400;  Epack = 47.6;

P = load('EV_DriveProfiles.mat');
S = load('EV_1800s_SyntheticDriveProfiles.mat');

names = {'Eco_ours','Normal_ours','Sport_ours','City_friends','Eco_friends','Sport_friends'};
prof  = cell(1,6);
prof{1} = fromDataset(P.EcoProfile);
prof{2} = fromDataset(P.NormalProfile);
prof{3} = fromDataset(P.SportProfile);
prof{4} = [S.CityTime(:)  S.CityVehicleSpeed(:)];
prof{5} = [S.EcoTime(:)   S.EcoVehicleSpeed(:)];
prof{6} = [S.SportTime(:) S.SportVehicleSpeed(:)];

n = numel(names);
MaxSpeed_kmh = zeros(n,1); AvgSpeed_kmh = zeros(n,1); Distance_km = zeros(n,1);
PeakAccel = zeros(n,1); PeakForce_N = zeros(n,1); PeakPmech_kW = zeros(n,1);
PeakPbatt_kW = zeros(n,1); PeakCurrent_A = zeros(n,1); PeakCellCurrent_A = zeros(n,1);
Energy_kWh = zeros(n,1); Wh_per_km = zeros(n,1); SOCused_pct = zeros(n,1);

figure('Name','Battery power demand','Position',[100 100 1200 800]);
tiledlayout(3,2);
for k = 1:n
    t = prof{k}(:,1);  v = prof{k}(:,2);
    a  = gradient(v, t);
    F  = m*a + kDrag*v.^2 + Frr;
    Pm = F.*v;
    Pb = max(Pm/eta, 0);
    I  = Pb/Vbat;

    MaxSpeed_kmh(k)      = max(v)*3.6;
    Distance_km(k)       = trapz(t, v)/1000;
    AvgSpeed_kmh(k)      = Distance_km(k)/(t(end)-t(1))*3600;
    PeakAccel(k)         = max(a);
    PeakForce_N(k)       = max(F);
    PeakPmech_kW(k)      = max(Pm)/1e3;
    PeakPbatt_kW(k)      = max(Pb)/1e3;
    PeakCurrent_A(k)     = max(I);
    PeakCellCurrent_A(k) = max(I)/30;
    Energy_kWh(k)        = trapz(t, Pb)/3.6e6;
    Wh_per_km(k)         = Energy_kWh(k)*1000/Distance_km(k);
    SOCused_pct(k)       = Energy_kWh(k)/Epack*100;

    nexttile; plot(t, Pb/1e3); grid on
    title(strrep(names{k},'_',' ')); xlabel('Time (s)'); ylabel('Battery power (kW)');
end
exportgraphics(gcf, 'vehicle_power_demand.png', 'Resolution', 200);

T = table(MaxSpeed_kmh, AvgSpeed_kmh, Distance_km, PeakAccel, PeakForce_N, ...
          PeakPmech_kW, PeakPbatt_kW, PeakCurrent_A, PeakCellCurrent_A, ...
          Energy_kWh, Wh_per_km, SOCused_pct, 'RowNames', names);
disp(T)
writetable(T, 'vehicle_analysis.xlsx', 'WriteRowNames', true);
fprintf('Saved vehicle_analysis.xlsx and vehicle_power_demand.png in %s\n', pwd);

function out = fromDataset(ds)
    x = ds.get('VehicleSpeed');
    if ~isa(x, 'timeseries'), x = x.Values; end
    out = [x.Time(:) x.Data(:)];
end