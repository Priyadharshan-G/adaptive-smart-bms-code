%% LFP full scope result exporter (PNG and editable FIG) — original project in D:\Electrical
% Save as export_LFP_results_FIXED.m; run from MATLAB Command Window.
% Requires MATLAB, Simulink, Simscape, Simscape Battery, model dependencies.
% Does NOT fabricate driving profiles, cell parameters, or numerical results.

clc;
projectDir = 'D:\Electrical';
modelName = 'ESSSimulation_LFP';
modelFile = fullfile(projectDir, [modelName '.slx']);
libName = 'ANR26650_108S57P_Library_lib';
libFile = fullfile(projectDir, [libName '.slx']);
packDataFile = fullfile(projectDir, 'ANR26650_108S57P_Library.mat');
outputDir = fullfile(projectDir, 'LFP_simulation_results');
figuresDir = fullfile(outputDir, 'Figures');

assert(isfile(modelFile), 'Model missing: %s',modelFile);
assert(isfile(libFile), 'Generated battery library missing: %s',libFile);
assert(isfile(packDataFile), 'Battery MAT file missing: %s',packDataFile);

% Locate EXISTING Signal Editor source; never silently substitute data.
driveFiles = dir(fullfile(projectDir, '**', 'EV_DriveProfiles.mat'));
if isfile(fullfile(projectDir, 'EV_DriveProfiles.mat'))
    driveFile = fullfile(projectDir, 'EV_DriveProfiles.mat');
elseif ~isempty(driveFiles)
    driveFile = fullfile(driveFiles(1).folder,driveFiles(1).name);
else
    [f,p] = uigetfile('*.mat','Locate EV_DriveProfiles.mat');
    assert(~isequal(f,0),'Select the original EV_DriveProfiles.mat to continue.');
    driveFile = fullfile(p,f);
    assert(strcmpi(f,'EV_DriveProfiles.mat'), ...
        'Selected file must be named EV_DriveProfiles.mat.');
end

% Add the project root so +package namespaces remain discoverable.
addpath(projectDir);
addpath(fileparts(driveFile));
rehash;
fprintf('Drive profiles: %s\n',driveFile);
fprintf('Battery library: %s\n',libFile);

% Generated Simscape Battery libraries typically use variables in their
% associated MAT file. Load explicitly into MATLAB base workspace.
packVariables = load(packDataFile);
packNames = fieldnames(packVariables);
for i = 1:numel(packNames)
    assignin('base',packNames{i},packVariables.(packNames{i}));
end
fprintf('Loaded %d battery data variable(s) into base workspace.\n',numel(packNames));

% Resolve model name conflicts without silently discarding edits.
if bdIsLoaded(modelName)
    oldPath = get_param(modelName,'FileName');
    if ~strcmpi(oldPath,modelFile)
        error(['A different copy is loaded: %s\nSave and close it with ' ...
            'bdclose(''%s''), then rerun this script.'], oldPath, modelName);
    end
end
load_system(libFile);
load_system(modelFile);

% Verify Signal Editor points to the correct dataset by ensuring the
% correctly named file is accessible. If block contains an absolute stale
% path, edit the Signal Editor block's file setting in Simulink.
assert(exist('EV_DriveProfiles.mat','file')==2, ...
    'Signal Editor dataset is not on the MATLAB path.');

if ~exist(outputDir,'dir'), mkdir(outputDir); end
if ~exist(figuresDir,'dir'), mkdir(figuresDir); end
set_param(modelName,'SignalLogging','on','SignalLoggingName','logsout', ...
    'ReturnWorkspaceOutputs','on','SaveTime','on');

% Log lines feeding ALL scopes anywhere inside this model, including
% top-level and subsystems. Include input port in unique log name.
scopes = find_system(modelName,'LookUnderMasks','all', ...
    'FollowLinks','on','BlockType','Scope');
map = struct('scope',{},'port',{},'signal',{},'status',{});
for i = 1:numel(scopes)
    h = get_param(scopes{i},'PortHandles');
    for j = 1:numel(h.Inport)
        sig = matlab.lang.makeValidName(sprintf('Scope_%03d_port_%02d',i,j));
        lineHandle = get_param(h.Inport(j),'Line');
        status = 'logged';
        if lineHandle == -1
            status='unconnected';
        else
            try
                set_param(lineHandle,'DataLogging','on', ...
                    'DataLoggingNameMode','Custom','DataLoggingName',sig);
            catch err
                status=sprintf('logging failed: %s',err.message);
            end
        end
        map(end+1)=struct('scope',scopes{i},'port',j, ...
            'signal',sig,'status',status); %#ok<SAGROW>
    end
end
save(fullfile(outputDir,'scope_signal_mapping.mat'),'map');

fprintf('Found %d Scope block(s); starting simulation...\n',numel(scopes));
try
    simOut = sim(modelName, 'StopTime',get_param(modelName,'StopTime'));
catch err
    fid=fopen(fullfile(outputDir,'SIMULATION_ERROR.txt'),'w');
    if fid>=0
        fprintf(fid,'%s\n\n%s\n',err.message, ...
            getReport(err,'extended','hyperlinks','off'));
        fclose(fid);
    end
    rethrow(err);
end

logs = simOut.logsout;
summary = table('Size',[0 7], ...
    'VariableTypes',{'string','string','double','double','double','double','double'}, ...
    'VariableNames',{'Scope','Signal','Initial','Final','Minimum','Maximum','Mean'});
exported = 0;
for k=1:numel(map)
    if ~strcmp(map(k).status,'logged'), continue; end
    try
        el=logs.getElement(map(k).signal);
        ts=el.Values;
        t=double(ts.Time(:));
        y=double(ts.Data);
        if isempty(t), continue; end
        y=reshape(y,numel(t),[]);
        [~,scopeLabel]=fileparts(map(k).scope);
        name=sprintf('%s_%s',map(k).signal, ...
            regexprep(scopeLabel,'[^A-Za-z0-9_]','_'));
        vars=[{'Time_s'},arrayfun(@(q)sprintf('Channel_%02d',q), ...
            1:size(y,2),'UniformOutput',false)];
        writetable(array2table([t,y],'VariableNames',vars), ...
            fullfile(outputDir,[name '.csv']));
        fig=figure('Visible','off','Color','w');
        plot(t,y,'LineWidth',1.2);
        grid on; xlabel('Time (s)'); ylabel(scopeLabel,'Interpreter','none');
        title(map(k).scope,'Interpreter','none');
        if size(y,2)>1 && size(y,2)<=12
            legend(vars(2:end),'Interpreter','none','Location','best');
        end
        exportgraphics(fig,fullfile(figuresDir,[name '.png']),'Resolution',300);
        savefig(fig,fullfile(figuresDir,[name '.fig']));
        close(fig);
        exported=exported+1;
        for c=1:size(y,2)
            v=y(:,c); valid=v(isfinite(v));
            if isempty(valid),continue;end
            summary(end+1,:)={string(map(k).scope), ...
                string(sprintf('%s_channel_%02d',map(k).signal,c)), ...
                v(1),v(end),min(valid),max(valid),mean(valid)}; %#ok<SAGROW>
        end
    catch err
        warning('Could not export %s: %s',map(k).scope,err.message);
    end
end
writetable(summary,fullfile(outputDir,'results_summary.csv'));
save(fullfile(outputDir,'simulation_output.mat'),'simOut','-v7.3');
fprintf('\nFINISHED: %d scope input(s) exported.\nFolder: %s\n',exported,outputDir);
fprintf('Scope count: %d. See mapping MAT and results_summary.csv.\n',numel(scopes));
fprintf('Saved PNG + FIG files in: %s\n',figuresDir);
