%% udp_stream_pack.m - stream any pack from MATLAB to the live BMS in dashboard_v2.py
% 1) In Python, once:      python live_udp.py export
%    (writes packs/NCR|LFP/<Mode>_<fresh|aged>_stream.csv; P45B uses the original files at the repo root)
% 2) In the dashboard:     Live UDP tab -> pick the same pack in the sidebar -> Start listening
% 3) Run this script.
% Packet: 7 doubles [seq, t, V_pack, I_pack, T, SOC_true, send_time], same as udp_stream.m.

pack    = "NCR";      % "P45B", "NCR" or "LFP"
mode    = "Sport";    % "Eco", "Normal" or "Sport"
aged    = true;
speedup = 20;
host    = "127.0.0.1";
port    = 5005;

root = fileparts(mfilename("fullpath"));
if pack == "P45B", folder = root; else, folder = fullfile(root, "packs", pack); end
if aged, cond = "aged"; else, cond = "fresh"; end
f = fullfile(folder, mode + "_" + cond + "_stream.csv");
if ~isfile(f), error("Missing %s - run  python live_udp.py export  first.", f); end
D = readtable(f, "VariableNamingRule", "preserve");
n = height(D);

sock = java.net.DatagramSocket();
addr = java.net.InetAddress.getByName(host);
fprintf("Streaming %s %s %s (%d samples) to %s:%d ...\n", pack, mode, cond, n, host, port);

dt = [0; diff(D.Time_s)];
for k = 1:n
    pause(dt(k) / speedup);
    pkt = [k, D.Time_s(k), D.Voltage_V(k), D.Current_A(k), D.Temp_degC(k), D.SOC_pct(k), ...
           posixtime(datetime("now", "TimeZone", "UTC"))];
    bytes = typecast(pkt, "int8");
    sock.send(java.net.DatagramPacket(bytes, numel(bytes), addr, port));
    if mod(k, 20) == 0, fprintf("  sent %d/%d\n", k, n); end
end
bytes = typecast([-1 0 0 0 0 0 posixtime(datetime("now", "TimeZone", "UTC"))], "int8");
sock.send(java.net.DatagramPacket(bytes, numel(bytes), addr, port));
sock.close();
disp("Stream finished.")
