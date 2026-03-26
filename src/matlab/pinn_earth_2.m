% =========================================================================
%  Physics-Informed Neural Network (PINN) — Earth Orbit
% =========================================================================
%
%  PURPOSE:
%    Extend the pure NN baseline by embedding two physical laws directly
%    into the loss function, computed via automatic differentiation.
%    The network is now penalised for violating Newtonian gravity and
%    angular momentum conservation — not just for fitting the data.
%
%  APPROACH:
%    - Input:  normalized time t ∈ [0, 1]
%    - Output: normalized 2D position (x, y) in AU
%    - Loss:   L_data  +  λ₁·L_Newton  +  λ₂·L_AngularMomentum
%    - Optimizer: Adam (lr = 1e-3)
%
%  ARCHITECTURE:
%    t (1)  >  FC(60) + tanh  >  FC(60) + tanh  >  (x, y) (2)
%
%  PHYSICS LOSSES:
%    L_Newton       — penalises violation of d²r/dt² = -GM·r/|r|³
%                     2nd derivatives computed via autodiff (dlgradient)
%                     with chain rule to convert normalised → AU/year²
%    L_AngularMom   — penalises violation of L = x·vy - y·vx = const
%                     velocities recovered from position output via autodiff
%
%  ADAPTIVE LOSS WEIGHTING:
%    λ₁ and λ₂ are ramped linearly from 0 to their target values between
%    epochs 2000–3500. The network first builds a data fit, then physics
%    constraints are gradually tightened. Starting with λ > 0 destabilises
%    early training.
%
%  LIMITATION:
%    Physics constraints are soft penalties — they can still be violated.
%    Competing objectives (data fit vs two physics terms) are hard to
%    balance and sensitive to the weight schedule. This motivates the
%    discrepancy modelling approach in Stage 3.
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
valid = ~isnan(data.Var3);
data  = data(valid, :);

% ========================== CONSTANTS =====================================
AU = 1.496e8;                       % km per AU
GM = 4 * pi^2;                      % AU³/year²  (GM in natural units)

% ========================== DATA EXTRACTION ================================
t  = data.Var1;                     % Julian Date (numeric)
x  = data.Var3;                     % X position [km]
y  = data.Var4;                     % Y position [km]
vx = data.Var5;                     % VX [km/s]
vy = data.Var6;                     % VY [km/s]

x_au = x / AU;
y_au = y / AU;

% ========================== NORMALISATION =================================
% Time: scale to [0, 1]
t0      = t(1);
tf      = t(end);
t_norm  = (t - t0) / (tf - t0);
t_scale = (tf - t0) / 365.25;      % Julian days → years (needed for chain rule)

% Positions: zero mean, unit std
x_au_mean = mean(x_au);  x_au_std = std(x_au);
y_au_mean = mean(y_au);  y_au_std = std(y_au);

x_norm = (x_au - x_au_mean) / x_au_std;
y_norm = (y_au - y_au_mean) / y_au_std;

% ========================== ANGULAR MOMENTUM REFERENCE ====================
% Compute L = x·vy - y·vx from NASA data — this is the conserved value
% the network must match
vx_au_yr = vx * (365.25 * 86400) / AU;   % km/s → AU/year
vy_au_yr = vy * (365.25 * 86400) / AU;
L_true   = x_au .* vy_au_yr - y_au .* vx_au_yr;
L_ref    = mean(L_true);                  % single conserved scalar

% ========================== NETWORK DEFINITION ============================
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(60)
    tanhLayer
    fullyConnectedLayer(2)              % output: [x_norm; y_norm]
];

net = dlnetwork(layers);

% Format as dlarrays (CB = Channel x Batch)
T  = dlarray(t_norm',            'CB');   % [1 x N]
XY = dlarray([x_norm'; y_norm'], 'CB');   % [2 x N]

% ========================== TRAINING ======================================
% 5000 epochs: first 2000 data-only, ramp physics in from 2000→3500,
% full physics constraints from 3500→5000
learnRate = 1e-3;
numEpochs = 2000;
avgG      = [];                     % Adam first moment
avgSqG    = [];                     % Adam second moment

monitor = trainingProgressMonitor( ...
    Metrics="Loss", Info="Epoch", XLabel="Iteration");

for epoch = 1:numEpochs

    % ---- Adaptive loss weights: linearly ramped from epoch 2000 to 3500 ----
    % Before epoch 2000: λ = 0 (pure data fit)
    % Epoch 2000–3500:   λ ramps up (physics gradually introduced)
    % After epoch 3500:  λ = max value (full physics constraints)
    P1_lambda = 1e-3 * min(1, max(0, (epoch - 2000) / 1500));   % Newton
    P2_lambda = 1e-3 * min(1, max(0, (epoch - 2000) / 1500));   % Ang. mom.

    [loss, grad] = dlfeval( ...
        @(n,T,XY) lossFn(n, T, XY, x_au_std, y_au_std, t_scale, GM, ...
                         P1_lambda, P2_lambda, L_ref), ...
        net, T, XY);

    [net, avgG, avgSqG] = adamupdate(net, grad, avgG, avgSqG, epoch, learnRate);

    % ---- Logging every 500 epochs ----
    if mod(epoch, 500) == 0
        XY_out = extractdata(forward(net, T));

        dataLoss = mean((XY_out - extractdata(XY)).^2, 'all');

        % Numerical physics residual (monitoring only)
        x_p = XY_out(1,:) * x_au_std;
        y_p = XY_out(2,:) * y_au_std;
        r_p = sqrt(x_p.^2 + y_p.^2);
        dt  = t_scale / length(t_norm);

        d2x_num  = diff(diff(x_p)) / dt^2;
        d2y_num  = diff(diff(y_p)) / dt^2;
        ax_req   = -GM * x_p(2:end-1) ./ r_p(2:end-1).^3;
        ay_req   = -GM * y_p(2:end-1) ./ r_p(2:end-1).^3;
        phys_res = mean((d2x_num - ax_req).^2 + (d2y_num - ay_req).^2, 'all');

        % Numerical angular momentum residual (monitoring only)
        vx_num     = diff(x_p) / dt;
        vy_num     = diff(y_p) / dt;
        L_num      = x_p(1:end-1) .* vy_num - y_p(1:end-1) .* vx_num;
        angMom_res = mean(((L_num - L_ref) / L_ref).^2, 'all');

        fprintf("Epoch %4d | Loss: %.6f | DataLoss: %.6f | PhysRes: %.4e | AngMomRes: %.6f | λ₁=%.1e\n", ...
            epoch, extractdata(loss), dataLoss, phys_res, angMom_res, P1_lambda);
    end

    recordMetrics(monitor, epoch, Loss=loss);
    updateInfo(monitor, Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * epoch / numEpochs;
end

% ========================== LOSS FUNCTION =================================
function [loss, grad] = lossFn(net, T, XY_true, x_au_std, y_au_std, ...
                                t_scale, GM, P1_lambda, P2_lambda, L_ref)

    XY_pred = forward(net, T);   % [2 x N]

    % ---- Data loss ----
    dataLoss = mean((XY_pred - XY_true).^2, 'all');

    % ---- 1st derivatives via autodiff (needed for both physics terms) ----
    % EnableHigherDerivatives keeps the tape alive for the 2nd pass (Newton)
    x_n   = XY_pred(1,:);
    y_n   = XY_pred(2,:);
    dx_dn = dlgradient(sum(x_n,'all'), T, 'EnableHigherDerivatives', true);
    dy_dn = dlgradient(sum(y_n,'all'), T, 'EnableHigherDerivatives', true);

    % Denormalise positions to AU for physics computations
    x_au_p = x_n * x_au_std;
    y_au_p = y_n * y_au_std;

    % ---- Physics loss: Newton's law of gravitation ----
    if P1_lambda > 0
        % 2nd derivatives w.r.t. t_norm
        d2x_dn2 = dlgradient(sum(dx_dn,'all'), T);
        d2y_dn2 = dlgradient(sum(dy_dn,'all'), T);

        % Chain rule: normalised → AU/year²
        % d²x_au/dt_yr² = (x_au_std / t_scale²) · d²x_n/dt_norm²
        d2x = (x_au_std / t_scale^2) * d2x_dn2;
        d2y = (y_au_std / t_scale^2) * d2y_dn2;

        r = sqrt(x_au_p.^2 + y_au_p.^2 + 1e-8);   % +1e-8 avoids divide-by-zero

        % Residual: how much does the trajectory violate Newton's law?
        Rx = d2x + GM * x_au_p ./ r.^3;
        Ry = d2y + GM * y_au_p ./ r.^3;

        physicsLoss = mean(Rx.^2 + Ry.^2, 'all');
    else
        physicsLoss = 0;
    end

    % ---- Angular momentum loss ----
    % Recover velocities from autodiff: normalised → AU/year
    vx_pred    = (x_au_std / t_scale) * dx_dn;
    vy_pred    = (y_au_std / t_scale) * dy_dn;
    L_pred     = x_au_p .* vy_pred - y_au_p .* vx_pred;
    angMomLoss = mean(((L_pred - L_ref) / L_ref).^2, 'all');

    % ---- Combined loss ----
    loss = dataLoss + P1_lambda * physicsLoss + P2_lambda * angMomLoss;
    grad = dlgradient(loss, net.Learnables);
end

% ========================== EVALUATION ====================================
XY_pred = extractdata(forward(net, T));

x_pred = (XY_pred(1,:) * x_au_std + x_au_mean)';
y_pred = (XY_pred(2,:) * y_au_std + y_au_mean)';

rmse = sqrt(mean((x_pred - x_au).^2 + (y_pred - y_au).^2));
fprintf("\nRMSE (PINN): %.4e AU\n", rmse);

% ========================== PLOT ==========================================
figure;
plot(x_au,   y_au,   'b.',  'MarkerSize', 4,   'DisplayName', 'NASA Data');
hold on;
plot(x_pred, y_pred, 'r--', 'LineWidth',  1.5, 'DisplayName', 'PINN Predicted');
plot(0, 0,           'y*',  'MarkerSize', 12,  'DisplayName', 'Sun');
legend('Location', 'best');
axis equal; grid on;
title(sprintf('Stage 2: PINN  (RMSE = %.2e AU)', rmse));
xlabel('X [AU]'); ylabel('Y [AU]');
