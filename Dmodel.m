%  DISCREPANCY MODELLING — Earth Orbit 

% ========================== DATA READING ==================================
data  = readtable("horizons_results.txt");
valid = ~isnan(data.Var3);
data  = data(valid, :);

% ========================== CONSTANTS =====================================
AU   = 1.496e8;
GM   = 4 * pi^2;        % AU^3/year^2
yr2s = 365.25 * 86400;

% ========================== DATA EXTRACTION ================================
t  = data.Var1;
x  = data.Var3 / AU;
y  = data.Var4 / AU;
% Var5 = Z position 
vx = data.Var6 * yr2s / AU;   % VX [km/s] -> AU/yr
vy = data.Var7 * yr2s / AU;   % VY [km/s] -> AU/yr

t_yr = (t - t(1)) / 365.25;
N    = length(t_yr);

% ========================== DIAGNOSTIC: CHECK ICs =========================
r0    = sqrt(x(1)^2 + y(1)^2);
v0    = sqrt(vx(1)^2 + vy(1)^2);
v_cir = sqrt(GM / r0);   % expected circular speed at this radius

fprintf("=== INITIAL CONDITIONS CHECK ===\n");
fprintf("  x0 = %.4f AU,  y0 = %.4f AU\n",  x(1), y(1));
fprintf("  vx0 = %.4f AU/yr,  vy0 = %.4f AU/yr\n", vx(1), vy(1));
fprintf("  |r0| = %.4f AU  (should be ~1.0)\n",  r0);
fprintf("  |v0| = %.4f AU/yr  (circular speed = %.4f AU/yr)\n", v0, v_cir);
fprintf("  Data: %d points over %.2f years\n\n", N, t_yr(end));

% ========================== STEP 1: KEPLERIAN BASELINE ====================
kepler_ode = @(~, s) [ s(3);
                        s(4);
                       -GM * s(1) / (s(1)^2 + s(2)^2)^1.5;
                       -GM * s(2) / (s(1)^2 + s(2)^2)^1.5 ];

ic   = double([x(1); y(1); vx(1); vy(1)]);   % ensure double
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
[~, sol_kep] = ode45(kepler_ode, t_yr, ic, opts);

x_kep  = sol_kep(:,1);
y_kep  = sol_kep(:,2);
vx_kep = sol_kep(:,3);
vy_kep = sol_kep(:,4);

% Quick sanity check — if Kepler fails, stop and report
rmse_kep_check = sqrt(mean((x_kep - x).^2 + (y_kep - y).^2));
fprintf("=== KEPLER SANITY CHECK ===\n");
fprintf("  RMSE Kepler vs NASA: %.4e AU\n", rmse_kep_check);
if rmse_kep_check > 0.1
    warning("Kepler RMSE is large (%.2f AU). Check units/ICs above.", rmse_kep_check);
    fprintf("  Hint: if |r0| is not ~1 AU or |v0| not ~6.28 AU/yr, there is a unit problem.\n\n");
end

% ========================== STEP 2: POSITION DISCREPANCY ==================
% Learn position residual directly: delta = NASA - Kepler
%
% avoid differentiating positions to estimate accelerations.
% With dt = 1/365 years, 2nd-differencing amplifies position noise by
% 1/dt^2 ~ 1e5

delta_x = x - x_kep;
delta_y = y - y_kep;

fprintf("\n=== DISCREPANCY CHECK ===\n");
fprintf("  Max |delta_x|: %.4e AU\n", max(abs(delta_x)));
fprintf("  Max |delta_y|: %.4e AU\n", max(abs(delta_y)));
fprintf("  Mean |delta|:  %.4e AU\n", mean(sqrt(delta_x.^2 + delta_y.^2)));
fprintf("  This is the position residual D must learn.\n\n");

% ========================== STEP 3: PREPARE NN DATA =======================
% Input  (state variables): [x_kep, y_kep, vx_kep, vy_kep]  — 4 features
% Output (discrepancy):     [delta_x, delta_y]               — 2 targets
%
% Using state variables as input (not time) means the correction depends
% on WHERE Earth is on its orbit 
% represents: D(x) not D(t).

state_raw  = [x_kep, y_kep, vx_kep, vy_kep];
s_mean     = mean(state_raw, 1);
s_std      = std(state_raw,  0, 1);
state_norm = (state_raw - s_mean) ./ s_std;   % zero mean, unit std

dx_scale   = std(delta_x);
dy_scale   = std(delta_y);
delta_norm = [delta_x / dx_scale,  delta_y / dy_scale];

%include atan2 as NN input
X_in  = dlarray(single(state_norm'), 'CB');   % [4 x N]
Y_tgt = dlarray(single(delta_norm'),  'CB');  % [2 x N]

% ========================== STEP 4: NETWORK (D block) =====================
% D: state (4) -> position correction (2)
layers = [
    featureInputLayer(4)
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(2)
];
net = dlnetwork(layers);
fprintf("Network (D block): 4 state inputs -> 32 -> 32 -> 2 position outputs\n");

% ========================== STEP 5: TRAINING ==============================
learnRate = 1e-3;
numEpochs = 5000;
avgG = []; avgSqG = [];

monitor = trainingProgressMonitor( ...
    Metrics="Loss", Info="Epoch", XLabel="Epoch");
yscale(monitor,"Loss","log")

fprintf("Training D block...\n");
for epoch = 1:numEpochs
    [loss, grad] = dlfeval(@discrepancyLoss, net, X_in, Y_tgt);
    [net, avgG, avgSqG] = adamupdate(net, grad, avgG, avgSqG, epoch, learnRate);

    if mod(epoch, 500) == 0
        fprintf("  Epoch %4d | Loss: %.6f\n", epoch, extractdata(loss));
    end

    recordMetrics(monitor, epoch, Loss=loss);
    updateInfo(monitor, Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * epoch / numEpochs;
end

% ========================== STEP 6: EVALUATE ==============================
% Final prediction: Kepler + D(Keplerian state)
delta_pred  = double(extractdata(forward(net, X_in)));
dx_pred     = delta_pred(1,:)' * dx_scale;
dy_pred     = delta_pred(2,:)' * dy_scale;

x_full = x_kep + dx_pred;
y_full = y_kep + dy_pred;
%% 

r_full = sqrt(x_full.^2 + y_full.^2);
r_kep = sqrt(x_kep.^2 + y_kep.^2);
r_data = sqrt(x.^2 + y.^2);

error_est = r_full - r_data;
error_kep = r_full - r_kep;

rmse_kep_1 = rms(error_kep);
rmse_est = rms(error_est);
fprintf("  RMSE Kepler :  %.4e AU\n", rmse_kep_1);
fprintf("  RMSE Est :  %.4e AU\n", rmse_est);
%% 

rmse_kep  = sqrt(mean((x_kep  - x).^2 + (y_kep  - y).^2));
rmse_full = sqrt(mean((x_full - x).^2 + (y_full - y).^2));

fprintf("\n=== RESULTS ===\n");
fprintf("  RMSE Kepler only:          %.4e AU\n", rmse_kep);
fprintf("  RMSE Kepler + D (hybrid):  %.4e AU\n", rmse_full);
fprintf("  Improvement factor:        %.1fx\n",   rmse_kep / rmse_full);

% ========================== PLOTS =========================================
figure('Name','Discrepancy Modelling','Position',[100 100 1200 900]);

% Plot 1: full orbit comparison
subplot(2,2,[1 2]);
plot(x,      y,      'bx',  'MarkerSize',8,   'DisplayName','NASA Data');
hold on;
plot(x_kep,  y_kep,  'g--', 'LineWidth',1.5,  'DisplayName','Kepler (physics only)');
plot(x_full, y_full, 'r--', 'LineWidth',1.5,  'DisplayName','Kepler + D (hybrid)');
plot(0, 0,           'y*',  'MarkerSize',14,   'DisplayName','Sun');
legend('Location','best'); axis equal; grid on;
title(sprintf('Discrepancy Modelling — Earth Orbit  (RMSE: Kepler=%.2e, Hybrid=%.2e AU)', ...
    rmse_kep, rmse_full));
xlabel('X [AU]'); ylabel('Y [AU]');

% Plot 2: X discrepancy — true vs learned
subplot(2,2,3);
plot(t_yr, delta_x * 1e3,  'b',   'LineWidth',1.2, 'DisplayName','True \Delta x (NASA - Kepler)');
hold on;
plot(t_yr, dx_pred * 1e3,  'r--', 'LineWidth',1.2, 'DisplayName','D block output');
legend; grid on;
xlabel('Time [years]'); ylabel('\Delta x  [×10^{-3} AU]');
title('X Position Discrepancy: True vs D block');

% Plot 3: Y discrepancy — true vs learned
subplot(2,2,4);
plot(t_yr, delta_y * 1e3,  'b',   'LineWidth',1.2, 'DisplayName','True \Delta y (NASA - Kepler)');
hold on;
plot(t_yr, dy_pred * 1e3,  'r--', 'LineWidth',1.2, 'DisplayName','D block output');
legend; grid on;
xlabel('Time [years]'); ylabel('\Delta y  [×10^{-3} AU]');
title('Y Position Discrepancy: True vs D block');

%plot(1:length(error_est), error_est, 1:length(error_est), error_kep)

% ========================== LOSS FUNCTION =================================
function [loss, grad] = discrepancyLoss(net, X, Y)
    Y_pred = forward(net, X);
    loss   = mean((Y_pred - Y).^2, 'all');
    grad   = dlgradient(loss, net.Learnables);
end