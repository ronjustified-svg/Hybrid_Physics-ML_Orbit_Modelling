% =========================================================================
%  Pure Neural Network Baseline — Earth Orbit
% =========================================================================
%
%  PURPOSE:
%    Fit a feedforward neural network to Earth's orbital trajectory using
%    NASA Horizons data. No physics is used — this is a pure data-driven
%    baseline to establish a reference point before introducing any
%    physics-informed or hybrid approaches.
%
%  APPROACH:
%    - Input:  normalized time t ∈ [0, 1]
%    - Output: normalized 2D position (x, y) in AU
%    - Loss:   mean squared error (MSE) on position
%    - Optimizer: Adam (lr = 1e-3)
%
%  ARCHITECTURE:
%    t (1)  >  FC(60) + tanh  >  FC(60) + tanh  >  (x, y) (2)
%
%  LIMITATION:
%    The network has no knowledge of gravity, conservation laws, or orbital
%    periodicity. It treats the orbit as an arbitrary curve-fitting problem
%    and will fail to generalise outside the training window.
%    This motivates the physics-informed and discrepancy modelling stages.
%
%  DATA:
%    NASA JPL Horizons — Earth ephemeris, 2023–2024
%    ~365 daily observations, heliocentric ecliptic J2000 frame
%
%  REQUIREMENTS:
%    MATLAB R2022b+, Deep Learning Toolbox
%    horizons_results.txt must be in the working directory
% =========================================================================

% ========================== DATA READING ==================================
data  = readtable("horizons_results.txt");
valid = ~isnan(data.Var3);          % remove footer rows with NaN positions
data  = data(valid, :);

% ========================== CONSTANTS =====================================
AU = 1.496e8;                       % km per AU

% ========================== DATA EXTRACTION ================================
t  = data.Var1;                     % Julian Date (numeric)
x  = data.Var3;                     % X position [km]
y  = data.Var4;                     % Y position [km]

x_au = x / AU;                      % convert to AU
y_au = y / AU;

% ========================== NORMALISATION =================================
% Time: scale to [0, 1] for stable training
t0     = t(1);
tf     = t(end);
t_norm = (t - t0) / (tf - t0);

% Positions: zero mean, unit std — keeps activations in a healthy range
x_au_mean = mean(x_au);  x_au_std = std(x_au);
y_au_mean = mean(y_au);  y_au_std = std(y_au);

x_norm = (x_au - x_au_mean) / x_au_std;
y_norm = (y_au - y_au_mean) / y_au_std;

% ========================== NETWORK DEFINITION ============================
% Simple 2-hidden-layer MLP with tanh activations
% tanh chosen over ReLU: smooth derivatives, bounded output, suited to
% continuous periodic signals like orbits
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(2)          % output: [x_norm; y_norm]
];

net = dlnetwork(layers);

% Format as dlarrays (CB = Channel x Batch)
T  = dlarray(t_norm',            'CB');   % [1 x N]
XY = dlarray([x_norm'; y_norm'], 'CB');   % [2 x N]

% ========================== TRAINING ======================================
learnRate = 1e-3;
numEpochs = 2000;
avgG      = [];                     % Adam first moment
avgSqG    = [];                     % Adam second moment

monitor = trainingProgressMonitor( ...
    Metrics="Loss", Info="Epoch", XLabel="Iteration");

for epoch = 1:numEpochs

    [loss, grad] = dlfeval(@lossFn, net, T, XY);
    [net, avgG, avgSqG] = adamupdate(net, grad, avgG, avgSqG, epoch, learnRate);

    if mod(epoch, 500) == 0
        fprintf("Epoch %4d | Loss: %.6f\n", epoch, extractdata(loss));
    end

    recordMetrics(monitor, epoch, Loss=loss);
    updateInfo(monitor, Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * epoch / numEpochs;
end

% ========================== LOSS FUNCTION =================================
% Standard MSE — no physics terms
function [loss, grad] = lossFn(net, T, XY_true)
    XY_pred = forward(net, T);
    loss    = mean((XY_pred - XY_true).^2, 'all');
    grad    = dlgradient(loss, net.Learnables);
end

% ========================== EVALUATION ====================================
XY_pred = extractdata(forward(net, T));

% Denormalise back to AU
x_pred = (XY_pred(1,:) * x_au_std + x_au_mean)';
y_pred = (XY_pred(2,:) * y_au_std + y_au_mean)';

rmse = sqrt(mean((x_pred - x_au).^2 + (y_pred - y_au).^2));
fprintf("\nRMSE (Pure NN): %.4e AU\n", rmse);

% ========================== PLOT ==========================================
figure;
plot(x_au,    y_au,    'b.',  'MarkerSize', 4,   'DisplayName', 'NASA Data');
hold on;
plot(x_pred,  y_pred,  'r--', 'LineWidth',  1.5, 'DisplayName', 'NN Predicted');
plot(0, 0,              'y*', 'MarkerSize',  12,  'DisplayName', 'Sun');
legend('Location', 'best');
axis equal; grid on;
title(sprintf('Stage 1: Pure NN Baseline  (RMSE = %.2e AU)', rmse));
xlabel('X [AU]'); ylabel('Y [AU]');
