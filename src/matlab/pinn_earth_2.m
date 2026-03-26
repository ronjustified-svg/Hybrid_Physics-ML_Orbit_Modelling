% ======================= Data Reading / CSV Horizons =======================
data = readtable("horizons_results.txt");
valid = ~isnan(data.Var3);
data = data(valid, :);

% ========================== CONSTANTS =====================================
AU = 1.496e8;   % km per AU
GM = 4 * pi^2;  % AU³/year²

% ========================== DATA EXTRACTION ================================
t  = data.Var1;   % Julian Date
x  = data.Var3;   % X position [km]
y  = data.Var4;   % Y position [km]
vx = data.Var5;   % VX [km/s]
vy = data.Var6;   % VY [km/s]

x_au = x / AU;
y_au = y / AU;

% Normalize time to [0, 1]
t0 = t(1);
tf = t(end);
t_norm = (t - t0) / (tf - t0);
t_scale = (tf - t0) / 365.25;   % Julian days → years

% Normalize positions to zero mean / unit std
x_au_mean = mean(x_au); x_au_std = std(x_au);
y_au_mean = mean(y_au); y_au_std = std(y_au);

x_norm = (x_au - x_au_mean) / x_au_std;
y_norm = (y_au - y_au_mean) / y_au_std;

% Angular momentum reference from NASA data
vx_au_yr = vx * (365.25 * 86400) / AU;   % km/s → AU/year
vy_au_yr = vy * (365.25 * 86400) / AU;
L_true = x_au .* vy_au_yr - y_au .* vx_au_yr;
L_ref  = mean(L_true);   % conserved scalar the network must match

% ========================== NEURAL NETWORK =================================
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(2)      % output: [x_norm; y_norm]
];

net = dlnetwork(layers);

% Inputs / targets as dlarrays  (CB = Channel x Batch)
T  = dlarray(t_norm',            'CB');   % [1 x N]
XY = dlarray([x_norm'; y_norm'], 'CB');   % [2 x N]

% ========================== TRAINING =======================================
learnRate = 1e-3;
numEpochs = 2000;
avgG      = [];
avgSqG    = [];

monitor = trainingProgressMonitor( ...
    Metrics="Loss", Info="Epoch", XLabel="Iteration");

for epoch = 1:numEpochs

    % ---- Adaptive loss weights: linear ramp from epoch 2000 to 3500 ----
    %P1_lambda = 1e-3 * min(1, max(0, (epoch - 2000) / 1500));   % Newton
    %P2_lambda = 1e-3 * min(1, max(0, (epoch - 2000) / 1500));   % Ang. mom.
    P1_lambda = 0;
    P2_lambda = 0;

    [loss, grad] = dlfeval( ...
        @(n,T,XY) lossFn(n, T, XY, x_au_std, y_au_std, t_scale, GM, ...
                         P1_lambda, P2_lambda, L_ref), ...
        net, T, XY);

    [net, avgG, avgSqG] = adamupdate(net, grad, avgG, avgSqG, epoch, learnRate);

    % ---- Logging every 500 epochs ----
    if mod(epoch, 500) == 0
        XY_out = extractdata(forward(net, T));

        dataLoss = mean((XY_out - extractdata(XY)).^2, 'all');

        % Numerical physics residual (monitoring only, not used in training)
        x_p = XY_out(1,:) * x_au_std;
        y_p = XY_out(2,:) * y_au_std;
        r_p = sqrt(x_p.^2 + y_p.^2);
        dt  = t_scale / length(t_norm);

        d2x_num = diff(diff(x_p)) / dt^2;
        d2y_num = diff(diff(y_p)) / dt^2;
        ax_req  = -GM * x_p(2:end-1) ./ r_p(2:end-1).^3;
        ay_req  = -GM * y_p(2:end-1) ./ r_p(2:end-1).^3;
        phys_res = mean((d2x_num - ax_req).^2 + (d2y_num - ay_req).^2, 'all');

        % Numerical angular momentum residual (monitoring only)
        vx_num    = diff(x_p) / dt;
        vy_num    = diff(y_p) / dt;
        L_num     = x_p(1:end-1) .* vy_num - y_p(1:end-1) .* vx_num;
        angMom_res = mean(((L_num - L_ref) / L_ref).^2, 'all');

        fprintf("Epoch %4d | Loss: %.6f | DataLoss: %.6f | PhysRes: %.4f | AngMomRes: %.6f\n", ...
            epoch, extractdata(loss), dataLoss, phys_res, angMom_res);
    end

    recordMetrics(monitor, epoch, Loss=loss);
    updateInfo(monitor, Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * epoch / numEpochs;
end

% ========================== LOSS FUNCTION ==================================
function [loss, grad] = lossFn(net, T, XY_true, x_au_std, y_au_std, ...
                                t_scale, GM, P1_lambda, P2_lambda, L_ref)

    XY_pred = forward(net, T);   % [2 x N]

    % ---- Data loss ----
    dataLoss = mean((XY_pred - XY_true).^2, 'all');

    % ---- 1st derivatives: always computed (needed for angular momentum) ----
    x_n = XY_pred(1,:);
    y_n = XY_pred(2,:);

    dx_dn = dlgradient(sum(x_n,'all'), T, 'EnableHigherDerivatives', true);
    dy_dn = dlgradient(sum(y_n,'all'), T, 'EnableHigherDerivatives', true);

    % Denormalize positions to AU
    x_au_p = x_n * x_au_std;
    y_au_p = y_n * y_au_std;

    % ---- Physics loss (Newton's law) ----
    if P1_lambda > 0
        % 2nd derivatives w.r.t. t_norm
        d2x_dn2 = dlgradient(sum(dx_dn,'all'), T);
        d2y_dn2 = dlgradient(sum(dy_dn,'all'), T);

        % Chain rule: normalized -> AU/year^2
        d2x = (x_au_std / t_scale^2) * d2x_dn2;
        d2y = (y_au_std / t_scale^2) * d2y_dn2;

        r = sqrt(x_au_p.^2 + y_au_p.^2 + 1e-8);

        Rx = d2x + GM * x_au_p ./ r.^3;
        Ry = d2y + GM * y_au_p ./ r.^3;

        physicsLoss = mean(Rx.^2 + Ry.^2, 'all');
    else
        physicsLoss = 0;
    end

    % ---- Angular momentum loss ----
    % Convert autodiff velocities: normalized -> AU/year
    vx_pred = (x_au_std / t_scale) * dx_dn;
    vy_pred = (y_au_std / t_scale) * dy_dn;

    L_pred     = x_au_p .* vy_pred - y_au_p .* vx_pred;
    angMomLoss = mean(((L_pred - L_ref) / L_ref).^2, 'all');

    % ---- Combined loss ----
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
