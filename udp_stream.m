%% udp_stream.m - stream pack data from MATLAB to the Python BMS over UDP
mode    = "Sport";
aged    = true;
speedup = 20;
host    = "127.0.0.1";
port    = 5005;

if aged, cond = "aged"; else, cond = "fresh"; end
f = fullfile("C:\ESS", mode + "_" + cond + "_stream.csv");
D = readtable(f, "VariableNamingRule", "preserve");
n = height(D);

sock = java.net.DatagramSocket();
addr = java.net.InetAddress.getByName(host);
fprintf("Streaming %s (%d samples)...\n", f, n);

dt = [0; diff(D.Time_s)];
for k = 1:n
    pause(dt(k) / speedup);
    pkt = [k, D.Time_s(k), D.Voltage_V(k), D.Current_A(k), D.Temp_degC(k), D.SOC_pct(k), posixtime(datetime("now", "TimeZone", "UTC"))];
    bytes = typecast(pkt, "int8");
    sock.send(java.net.DatagramPacket(bytes, numel(bytes), addr, port));
    if mod(k, 20) == 0
        fprintf("  sent %d/%d\n", k, n);
    end
end

bytes = typecast([-1 0 0 0 0 0 posixtime(datetime("now", "TimeZone", "UTC"))], "int8");
sock.send(java.net.DatagramPacket(bytes, numel(bytes), addr, port));
sock.close();
disp("Stream finished.")