
% ======================= Data Reading/ CSV Horizon =======================
data = readtable("horizons_results.txt");
% Remove rows with NaN in position columns (the footer rows)
valid = ~isnan(data.Var3);
data = data(valid, :);

% CONSTANTS ===============================================================
AU = 1.496e8;  % km per AU

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

% Target: [x; y] as (2 x N) dlarray
XY = dlarray([x_norm'; y_norm'], 'CB');

learnRate = 1e-3;
numEpochs = 2000;
avgG   = [];   % first moment
avgSqG = [];   % second moment

numIterations = numEpochs;

%monitor = trainingProgressMonitor( ...
 %   Metrics="Loss", ...
  %  Info=["Epoch"], ...
   % XLabel="Iteration");

iteration  = 0;

for epoch = 1:numEpochs
    iteration = iteration + 1;
    [loss, grad] = dlfeval(@lossFn, net, T, XY);
    [net, avgG, avgSqG] = adamupdate(net, ...
                                    grad, ...
                                    avgG, ...
                                    avgSqG, ...
                                    epoch, ...
                                    learnRate);

    if mod(epoch, 500) == 0
        fprintf("Epoch %d | Loss: %.6f\n", epoch, extractdata(loss));
    end

    % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
end

% loss function , IP: NN, T, XY True , OP: loss, grad
function [loss, grad] = lossFn(net, T, XY_true)
    XY_pred = forward(net, T);
    loss = mean((XY_pred - XY_true).^2, 'all');
    grad = dlgradient(loss, net.Learnables);
end

XY_pred = extractdata(forward(net, T));

% Denormalize — use the same variables from normalization

x_pred = (XY_pred(1,:) * x_au_std + x_au_mean)';   % [366x1]
y_pred = (XY_pred(2,:) * y_au_std + y_au_mean)';   % [366x1]

error_x = x' - XY_pred(1,:);
error_y = y' - XY_pred(2,:);

% ============================= Figures / PLOT ============================

% Plot
figure;
plot3(t_norm, x_au, y_au, 'b.', 'Markersize', 2); hold on;
plot3(t_norm, x_pred, y_pred, 'r--', 'LineWidth', 1.5);
legend('Data Orbit', 'PINN Predicted');
axis equal; grid on;
title('Earth Orbit: Data vs NN Predicted');
xlabel('X [AU]'); ylabel('Y [AU]');

