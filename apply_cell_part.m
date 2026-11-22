function apply_cell_part(refCellBlock, packBlock)
%APPLY_CELL_PART Copy manufacturer cell data from a Battery Equivalent Circuit
% block (RefCell, with a part applied via "Select part") into every module of
% a Battery Builder pack.
%
%   apply_cell_part('ESSSimulation_NCR/RefCell', 'ESSSimulation_NCR/NCR18650BD_98S45P')
%   apply_cell_part('ESSSimulation_LFP/RefCell', 'ESSSimulation_LFP/ANR26650_120S52P')
%
% Name mapping (R2025a):
%   RefCell (batt.cells.batteryEquivalentCircuit)   ->  pack module (ModuleType1)
%   BatteryCapacity                                 ->  AHCell
%   SOCBreakpoints                                  ->  SOC_vecCell
%   OpenCircuitVoltage  (or 25 degC column of OpenCircuitVoltageThermal) -> V0_vecCell
%   R0                  (or 25 degC column of R0Thermal)                 -> R0_vecCell
%   VoltageRange                                    ->  V_rangeCell
% The module model has no temperature dependence, so the 25 degC data is used
% (the whole project runs at a constant 25 degC).

    Tuse = 25;                                               % degC
    mdl  = bdroot(packBlock);
    if getSimulinkBlockHandle(refCellBlock) == -1, error('Block not found: %s', refCellBlock); end
    if getSimulinkBlockHandle(packBlock)    == -1, error('Block not found: %s', packBlock);    end

    % ---------- read the reference cell ----------
    rp   = fieldnames(get_param(refCellBlock, 'DialogParameters'));
    has  = @(n) any(strcmp(rp, n));
    num  = @(n) evalin('base', get_param(refCellBlock, n));
    unit = @(n) unitOf(refCellBlock, n);

    capAh   = num('BatteryCapacity');   capU = unit('BatteryCapacity');
    soc     = num('SOCBreakpoints');    soc  = soc(:)';
    thermal = get_param(refCellBlock, 'ThermalModel');
    useThermal = ~contains(lower(thermal), 'constant') && has('OpenCircuitVoltageThermal');

    Tbp = [];
    if has('TemperatureBreakpoints'), Tbp = toDegC(num('TemperatureBreakpoints'), unit('TemperatureBreakpoints')); end

    % OCV vs SOC at 25 degC
    if useThermal
        V0 = atTemp(num('OpenCircuitVoltageThermal'), numel(soc), Tbp, Tuse);  V0U = unit('OpenCircuitVoltageThermal');
    else
        V0 = num('OpenCircuitVoltage');   V0U = unit('OpenCircuitVoltage');
    end
    V0 = V0(:)';

    % R0 vs SOC at 25 degC
    rsoc = soc;
    if has('ResistanceSOCBreakpoints')
        try, r = num('ResistanceSOCBreakpoints'); if numel(r) > 1, rsoc = r(:)'; end, catch, end
    end
    rT = Tbp;
    if has('ResistanceTemperatureBreakpoints')
        try, rT = toDegC(num('ResistanceTemperatureBreakpoints'), unit('ResistanceTemperatureBreakpoints')); catch, end
    end
    if useThermal && has('R0Thermal')
        R0 = num('R0Thermal');  R0U = unit('R0Thermal');
        if ~isvector(R0) || numel(R0) == numel(rT), R0 = atTemp(R0, numel(rsoc), rT, Tuse); end
    else
        R0 = num('R0');         R0U = unit('R0');
    end
    R0 = R0(:)';
    if isscalar(R0)
        R0 = repmat(R0, size(soc));
    elseif numel(R0) ~= numel(soc)
        R0 = interp1(rsoc, R0, soc, 'linear', 'extrap');     % put R0 on the OCV SOC grid
    end

    Vrange = num('VoltageRange');  VrU = unit('VoltageRange');

    if numel(V0) ~= numel(soc)
        error('OCV has %d points but SOC has %d.', numel(V0), numel(soc));
    end
    if abs(capAh - 27) < 1e-9
        warning(['RefCell capacity is 27 (the block default). The part was probably not applied: ' ...
                 'double-click RefCell > Select part > pick the cell > Apply, then run this again.']);
    end

    % ---------- write into every module ----------
    mods = find_system(packBlock, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                       'RegExp', 'on', 'ComponentPath', '.+');
    mods = mods(cellfun(@(b) isfield(get_param(b,'DialogParameters'), 'AHCell'), mods));
    if isempty(mods), error('No pack modules with AHCell found under %s.', packBlock); end

    for m = 1:numel(mods)
        b = mods{m};
        set_param(b, 'AHCell',      mat2str(capAh, 10));   trySet(b, 'AHCell_unit',      capU);
        set_param(b, 'SOC_vecCell', mat2str(soc, 10));     trySet(b, 'SOC_vecCell_unit', '1');
        set_param(b, 'V0_vecCell',  mat2str(V0, 10));      trySet(b, 'V0_vecCell_unit',  V0U);
        set_param(b, 'R0_vecCell',  mat2str(R0, 10));      trySet(b, 'R0_vecCell_unit',  R0U);
        set_param(b, 'V_rangeCell', mat2str(Vrange, 10));  trySet(b, 'V_rangeCell_unit', VrU);
        trySet(b, 'socCell', '1');                          % start full
    end
    set_param(mdl, 'Dirty', 'on');

    % ---------- report ----------
    P  = evalin('base', get_param(mods{1}, 'P'));
    S  = evalin('base', get_param(mods{1}, 'S'));
    Ns = S * numel(mods);
    capAhA = capAh; if contains(capU, 'mA'), capAhA = capAh/1000; end
    R0mOhm = R0 * 1000; if contains(R0U, 'mOhm'), R0mOhm = R0; end
    [~, i50] = min(abs(soc - 0.5));
    fprintf('\nUpdated %d modules of %s\n', numel(mods), packBlock);
    fprintf('Data used      : %s tables%s\n', ternary(useThermal,'temperature-dependent','constant-temperature'), ...
            ternary(useThermal, sprintf(' at %d degC', Tuse), ''));
    fprintf('Cell           : %.3f %s | OCV %.3f V (0%%) / %.3f V (50%%) / %.3f V (100%%) | R0 at 50%% SOC %.1f mOhm\n', ...
            capAh, capU, V0(1), V0(i50), V0(end), R0mOhm(i50));
    fprintf('Pack %dS%dP    : %.1f Ah | expected starting voltage %.1f V\n', Ns, P, capAhA*P, Ns*V0(end));
    fprintf('Done. Delete RefCell, save the model (Ctrl+S), then run it.\n');
end

% ---------- helpers ----------
function v = atTemp(M, nSoc, T, Tuse)
    % pick the column (or row) of an SOC x T table at the temperature nearest Tuse
    if isvector(M) || isempty(T), v = M; return, end
    [~, iT] = min(abs(T(:) - Tuse));
    if size(M,1) == nSoc,      v = M(:, iT);
    elseif size(M,2) == nSoc,  v = M(iT, :);
    else, error('Table size %s does not match %d SOC points.', mat2str(size(M)), nSoc);
    end
end

function T = toDegC(T, u)
    if any(strcmpi(u, {'K','kelvin'})), T = T - 273.15; end
end

function u = unitOf(blk, n)
    try, u = get_param(blk, [n '_unit']); catch, u = ''; end
end

function trySet(blk, name, val)
    if isempty(val), return, end
    try, set_param(blk, name, val); catch, end
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
