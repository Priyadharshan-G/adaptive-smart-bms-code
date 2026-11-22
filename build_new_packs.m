%% build_new_packs.m
% Builds two extra 400 V-class EV packs for the Adaptive Smart BMS project,
% sized to the same ~47-48 kWh as the original P45B 98S30P pack.
%
%   Pack A (existing) : Molicel INR-21700-P45B  4.5 Ah  3.6 V  98S30P   (reference)
%   Pack B (new)      : Panasonic NCR18650BD    3.03 Ah 3.6 V  98S45P   (NCA, 18650)
%   Pack C (new)      : A123 ANR26650M1 (LFP)   2.3 Ah  3.3 V  120S52P  (LFP, 26650)
%
% Both new cells are in the Simscape Battery part collection in R2025a, so the
% OCV and resistance tables come from MathWorks' manufacturer-based data
% (Block Parameterization Manager) instead of hand-typed placeholder tables.
%
% Run from D:\Electrical. Needs MATLAB R2025a + Simscape Battery.
% Output: NCR18650BD_98S45P_Library_lib.slx + package folder
%         ANR26650_120S52P_Library_lib.slx  + package folder
% Keep the generated +<name> package folders in Git - the original pack lost
% its component files and had to be rebuilt by hand.

clear; clc;
projectDir = fileparts(mfilename('fullpath'));   % the folder this script is saved in (D:\Electrical)
cd(projectDir);
addpath(projectDir);
fprintf("Working in: %s\n", projectDir);

specs = [ ...
    packSpec("NCR18650BD_98S45P", "Panasonic NCR18650BD", ...
             65, 9,  0.048, 3.03, 3.6, 4.2, 2.5, 45,  7, 2, 7)   % 7S45P modules x2 x7 = 98S
    packSpec("ANR26650_120S52P",  "A123 ANR26650M1 (LFP)", ...
             65, 13, 0.076, 2.3,  3.3, 3.6, 2.0, 52, 10, 2, 6)   % 10S52P modules x2 x6 = 120S
];
% Masses are approximate (check each datasheet before quoting them in the paper).

ref = struct("name","P45B_98S30P (existing)","Ns",98,"Np",30, ...
             "capAh",4.493,"vNom",3.6,"vMax",4.2,"vMin",2.5,"massKg",0.070);

for k = 1:numel(specs)
    s  = specs(k);
    Ns = s.cellsPerModule * s.modulesPerAssembly * s.numAssemblies;
    fprintf("\n=== Building %s  (%dS%dP, %s) ===\n", s.name, Ns, s.Np, s.part);

    % Cell
    geom = batteryCylindricalGeometry(simscape.Value(s.heightMM,"mm"), ...
                                      simscape.Value(s.radiusMM,"mm"));
    c = batteryCell(geom);
    c.Mass = simscape.Value(s.massKg,"kg");

    % Parallel assembly -> module -> module assembly -> pack
    pa = batteryParallelAssembly(c, s.Np);
    m  = batteryModule(pa, s.cellsPerModule);
    m.ModelResolution = "Lumped";      % one equivalent circuit per module, same as Pack A
    ma = batteryModuleAssembly(repmat(m, 1, s.modulesPerAssembly));
    pk = batteryPack(repmat(ma, 1, s.numAssemblies));
    pk.Name = s.name;                  % block shows up with this name in the library

    % Optional 3-D view (useful screenshot for the paper)
    try
        f = uifigure("Name", s.name);
        batteryChart(f, pk);
    catch
        % batteryChart not available - skip the picture
    end

    % Generate the Simscape library
    libName = s.name + "_Library";
    % NumericValues puts real numbers in the block dialog (not workspace
    % variable names), so apply_cell_part.m can overwrite them directly.
    try
        buildBattery(pk, "LibraryName", libName, "Directory", pwd, ...
                     "MaskParameters", "NumericValues", ...
                     "MaskInitialTargets", "NumericValues");
    catch
        buildBattery(pk, "LibraryName", libName, "Directory", pwd);
    end
    fprintf("Built library: %s_lib.slx\n", libName);

    summarise(s.name, Ns, s.Np, s.capAh, s.vNom, s.vMax, s.vMin, s.massKg);
end

fprintf("\n=== Reference ===\n");
summarise(ref.name, ref.Ns, ref.Np, ref.capAh, ref.vNom, ref.vMax, ref.vMin, ref.massKg);

fprintf("\nNext: put each pack block in a copy of ESSSimulation.slx, then apply the\n");
fprintf("cell part with apply_cell_part.m (see the comments in that file).\n");

%% ---------- local functions ----------
function s = packSpec(name, part, heightMM, radiusMM, massKg, capAh, vNom, vMax, vMin, ...
                      Np, cellsPerModule, modulesPerAssembly, numAssemblies)
    s = struct("name",name,"part",part,"heightMM",heightMM,"radiusMM",radiusMM, ...
               "massKg",massKg,"capAh",capAh,"vNom",vNom,"vMax",vMax,"vMin",vMin, ...
               "Np",Np,"cellsPerModule",cellsPerModule, ...
               "modulesPerAssembly",modulesPerAssembly,"numAssemblies",numAssemblies);
end

function summarise(name, Ns, Np, capAh, vNom, vMax, vMin, massKg)
    nCells = Ns*Np;
    Ah     = Np*capAh;
    kWh    = Ns*vNom*Ah/1000;
    fprintf("%-28s %dS%dP | %5d cells | %6.1f Ah | Vnom %5.1f V | Vmax %5.1f V | Vmin %5.1f V | %5.1f kWh | cells ~%4.0f kg | Sport peak %.2f A/cell\n", ...
        name, Ns, Np, nCells, Ah, Ns*vNom, Ns*vMax, Ns*vMin, kWh, nCells*massKg, 120.5/Np);
end
