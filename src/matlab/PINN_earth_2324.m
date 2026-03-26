
% ======================= Data Reading/ CSV Horizon =======================
data = readtable("horizons_results.txt");
% Remove rows with NaN in position columns (the footer rows)
valid = ~isnan(data.Var3);
data = data(valid, :);

% CONSTANTS ===============================================================
AU = 1.496e8;  % km per AU
GM = 4 * pi^2; % AU³/year²

% Data extraction
t       = data.Var1;          % Julian Date (numeric)
dates   = data.Var2;          % Date strings (cell array, need curly brace indexing), dates{1} = 'A.D. 2023-Jan-01 00:00:00.0000'
x       = data.Var3;          % X position [km]
y       = data.Var4;          % Y position [km]
vx      = data.Var5;          % VX [km/s]
vy      = data.Var6;          % VY [km/s]
vz      = data.Var7;          % VZ [km/s]

x_au = x / AU;
y_au = y / AU;

% Normalize time to [0, 1] 
t0 = t(1);
tf = t(end);
t_norm = (t - t0) / (tf - t0);

% Normalize positions to [-1, 1]
x_au_mean = mean(x_au); x_au_std = std(x_au);
y_au_mean = mean(y_au); y_au_std = std(y_au);

x_norm = (x_au - x_au_mean) / x_au_std;
y_norm = (y_au - y_au_mean) / y_au_std;

% Angular velocity conversion
vx_au_yr = vx * (365.25 * 86400) / AU;   % km/s to AU/year
vy_au_yr = vy * (365.25 * 86400) / AU;

L_true = x_au .* vy_au_yr - y_au .* vx_au_yr;  % [N×1], should be ~constant
L_ref = mean(L_true);   % the single conserved value we target

% =========================== NEURAL NETWORK ==============================

layers = [
    featureInputLayer(1)        % input: normalized time t
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(2)      % output: [x, y]
];

net = dlnetwork(layers);

% Input: t_norm as (1 x N) dlarray
T = dlarray(t_norm', 'CB');        % t_norm' is [1 X 366] i.e 1 feature(time), 366 batch samples(observations)
XY = dlarray([x_norm'; y_norm'], 'CB'); % Target: [x; y] as (2 x N) dlarray

% ==== TRAINING LOOP ======

learnRate = 1e-3;
numEpochs = 5000;
avgG   = [];   % first moment
avgSqG = [];   % second moment

numIterations = numEpochs;
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch", XLabel="Iteration");  

for epoch = 1:numEpochs
    
    % ---- Adaptive loss weights: linear ramp from epoch 2000 to 3500 ----
    P1_lambda = 1e-3 * min(1, max(0, (epoch - 2000) / 1500));   % Newton
    P2_lambda = 1e-2 * min(1, max(0, (epoch - 2000) / 1500));   % Ang. mom.

    [loss, grad] = dlfeval( ...
        @(n,T,XY) lossFn(n, T, XY, x_au_std, y_au_std, t_scale, GM, P1_lambda, P2_lambda, L_ref), ...
        net, T, XY);
    [net, avgG, avgSqG] = adamupdate(net, ...
                                    grad, ...
                                    avgG, ...
                                    avgSqG, ...
                                    epoch, ...
                                    learnRate);

    if mod(epoch, 500) == 0
        XY_out  = extractdata(forward(net, T));
        
        % Denormalize for physics residual check
        x_p = XY_out(1,:) * x_au_std;
        y_p = XY_out(2,:) * y_au_std;
        r_p = sqrt(x_p.^2 + y_p.^2);
        dt  = t_scale / length(t_norm);          % years per step

        d2x_num = diff(diff(x_p)) / dt^2;
        d2y_num = diff(diff(y_p)) / dt^2;
        ax_req  = -GM * x_p(2:end-1) ./ r_p(2:end-1).^3;
        ay_req  = -GM * y_p(2:end-1) ./ r_p(2:end-1).^3;
        phys_res = mean((d2x_num - ax_req).^2 + (d2y_num - ay_req).^2, 'all');
    
        dataLoss = mean((XY_out - extractdata(XY)).^2, 'all');

        fprintf("Epoch %4d | Loss: %.6f | DataLoss: %.6f | PhysRes: %.4f\n | PhysRes: %.4f\n", ...
            epoch, extractdata(loss), dataLoss, phys_res, a);

    end

    recordMetrics(monitor, epoch, Loss=loss);
    updateInfo(monitor, Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * epoch / numEpochs;

end

% ====== LOSS FUNCTION  =================

function [loss, grad] = lossFn(net, T, XY_true, x_au_std, y_au_std, t_scale, GM, P1_lambda, P2_lambda, L_ref)
    
    XY_pred = forward(net, T); % [2 x N]
    
    % ---- Data Loss ----
    dataLoss = mean((XY_pred - XY_true).^2, 'all');
    
    x_n = XY_pred(1,:);   % normalized x
    y_n = XY_pred(2,:);   % normalized y
    % 1st derivatives w.r.t t_norm
    % EnableHigherDerivatives keeps tape alive for 2nd pass
    dx_dn = dlgradient(sum(x_n, 'all'), T, 'EnableHigherDerivatives', true);
    dy_dn = dlgradient(sum(y_n, 'all'), T, 'EnableHigherDerivatives', true);
    
    % Denormalize positions to AU
    x_au_p = x_n * x_au_std;
    y_au_p = y_n * y_au_std;

    % ---- Physics Loss ----
    if P1_lambda > 0
         % 2nd derivatives w.r.t t_norm
        d2x_dn2 = dlgradient(sum(dx_dn, 'all'), T);
        d2y_dn2 = dlgradient(sum(dy_dn, 'all'), T);
    
        % Chain rule: convert from normalized units to AU/year²
        % d²x_au/dt_yr² = (x_au_std / t_scale²) * d²x_n/dt_norm²
        d2x = (x_au_std / t_scale^2) * d2x_dn2;
        d2y = (y_au_std / t_scale^2) * d2y_dn2;
        
        % Denormalize positions for r (needs to be in AU)
        x_au_p = x_n * x_au_std;
        y_au_p = y_n * y_au_std;
        r = sqrt(x_au_p.^2 + y_au_p.^2 + 1e-8);
    
        % Residuals — how much Newton's law is violated
        Rx = d2x + GM * x_au_p ./ r.^3;
        Ry = d2y + GM * y_au_p ./ r.^3;
    
        physicsLoss = mean(Rx.^2 + Ry.^2, 'all');
    else
        physicsLoss = 0;
    end
    
    % vx_pred, vy_pred from autodiff (in AU/year)
    vx_pred = (x_au_std / t_scale) * dx_dn;
    vy_pred = (y_au_std / t_scale) * dy_dn;
    
    L_pred = x_au_p .* vy_pred - y_au_p .* vx_pred;
    angMomLoss = mean((L_pred - L_ref).^2, 'all');

    % ---- Combined Loss ----
    loss = dataLoss + P1_lambda * physicsLoss + P2_lambda * angMomLoss;

    grad = dlgradient(loss, net.Learnables);
end

% ========================== PLOT ===========================================
XY_pred = extractdata(forward(net, T));

x_pred = (XY_pred(1,:) * x_au_std + x_au_mean)';
y_pred = (XY_pred(2,:) * y_au_std + y_au_mean)';

figure;
plot(x_au, y_au, 'b.', 'MarkerSize', 3); hold on;
plot(x_pred, y_pred, 'r--', 'LineWidth', 1.5);
plot(0, 0, 'y*', 'MarkerSize', 12);   % Sun at origin
legend('NASA Data', 'PINN Predicted', 'Sun');
axis equal; grid on;
title('Earth Orbit: Data vs PINN');
xlabel('X [AU]'); ylabel('Y [AU]');
