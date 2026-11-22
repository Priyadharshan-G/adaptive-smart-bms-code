%% build_lfp_pack.m - rebuilds the LFP pack as 108S57P (replaces 120S52P).
% A123 ANR26650M1 part data has OCV = 3.846 V at 100 % SOC, so 120S started at
% 461.5 V. 108S starts at ~415 V (400 V class, like P45B 408.5 V / NCR 402 V)
% and 57P keeps the energy at ~47.5 kWh.
% Layout: 9S57P modules x 2 per assembly x 6 assemblies = 108S57P.

projectDir = fileparts(mfilename('fullpath'));
cd(projectDir);

name = "ANR26650_108S57P";
geom = batteryCylindricalGeometry(simscape.Value(65,"mm"), simscape.Value(13,"mm"));
c = batteryCell(geom);
c.Mass = simscape.Value(0.076,"kg");

pa = batteryParallelAssembly(c, 57);
m  = batteryModule(pa, 9);
m.ModelResolution = "Lumped";
ma = batteryModuleAssembly(repmat(m, 1, 2));
pk = batteryPack(repmat(ma, 1, 6));
pk.Name = name;

try
    f = uifigure("Name", name);
    batteryChart(f, pk);
catch
end

libName = name + "_Library";
try
    buildBattery(pk, "LibraryName", libName, "Directory", pwd, ...
                 "MaskParameters", "NumericValues", "MaskInitialTargets", "NumericValues");
catch
    buildBattery(pk, "LibraryName", libName, "Directory", pwd);
end
fprintf("\nBuilt %s_lib.slx  (108S57P, 12 modules of 9S57P)\n", libName);
fprintf("Expected with A123 data: 133.4 Ah | start ~415 V | ~47.5 kWh\n");
