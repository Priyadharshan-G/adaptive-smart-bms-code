   S = load('EV_1800s_SyntheticDriveProfiles.mat');
   EcoProfile    = mkScen(S.CityTime,  S.CityVehicleSpeed);    % Traffic
   NormalProfile = mkScen(S.EcoTime,   S.EcoVehicleSpeed);     % Highway
   SportProfile  = mkScen(S.SportTime, S.SportVehicleSpeed);   % Hill
   save('EV_DriveProfiles.mat', 'EcoProfile', 'NormalProfile', 'SportProfile');
   disp('EV_DriveProfiles.mat now holds the dataset-based Eco/Normal/Sport profiles');

   function ds = mkScen(t, v)
       ts = timeseries(v(:), t(:));
       ts.Name = 'VehicleSpeed';  ts.DataInfo.Units = 'm/s';
       ds = Simulink.SimulationData.Dataset;
       ds = ds.addElement(ts, 'VehicleSpeed');
   end