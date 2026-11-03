%% BCP 2D - Main Convergence & Morphology Visualization Script (Classic Scaling)
%% ENGwCOM -Rev 1
tic;
clear; close all;

%% 1. GENERAL PARAMETERS
N       = 256;
M1      = 1;
epsilon = 0.1;
alpha   = 0.1; % Non-zero alpha for physical BCP microphase separation

t_smooth   = 1.4;  % Time to evolve random noise into a smooth profile
t_duration = 1.00;  % Integration interval for convergence tracking

% --- CONTROL FLAG FOR REFERENCE SOLUTION ---
compute_ref = true;  % Set to FALSE after the first run to load U_ref from disk instantly!
% --------------------------------------------

%% 2. GENERATE THE SMOOTH FUNCTION INITIAL CONDITION (U_start)
fprintf('Step 1: Evolving random noise to t = %.1f to obtain smooth state...\n', t_smooth);
dt_init = 0.01;
[U_start, TheUbar] = BCP2D_pre_evolve(dt_init, M1, t_smooth, N, epsilon, alpha);

%% 3. COMPUTE OR LOAD THE HIGH-RESOLUTION REFERENCE SOLUTION (U_ref)
if compute_ref
    fprintf('\nStep 2: Computing high-resolution reference solution with fine dt (1e-5)...\n');
    dt_ref = 1e-5; 
    U_ref  = BCP2D_run_segment(U_start, dt_ref, M1, t_duration, N, epsilon, alpha, TheUbar);
    save('BCP_Reference_Data.mat', 'U_ref');
    fprintf('        -> Reference solution saved successfully to BCP_Reference_Data.mat\n');
else
    fprintf('\nStep 2: Skipping reference computation. Loading U_ref from disk...\n');
    if x_exist('BCP_Reference_Data.mat', 'file')
        load('BCP_Reference_Data.mat', 'U_ref');
    else
        error('BCP_Reference_Data.mat not found! Please set compute_ref = true for this run.');
    end
end

%% 4. EXECUTE TEMPORAL REFINEMENT LOOP (9 STEPS)
fprintf('\nStep 3: Running temporal refinement loop across 9 time steps...\n');
dt_array  = 0.025 ./ (2.^(0:8)); 
l2_errors = zeros(size(dt_array));

M_dom = 2; a = 0; b = M_dom*pi; h = (b-a)/N;
x = linspace(a, b-h, N); [X, Y] = meshgrid(x, x);

for j = 1:length(dt_array)
    dt_current = dt_array(j);
    fprintf('  -> Running refinement step %d/9 with dt = %.7f (Exactly %d steps)\n', ...
            j, dt_current, round(t_duration/dt_current));
    
    U_current = BCP2D_run_segment(U_start, dt_current, M1, t_duration, N, epsilon, alpha, TheUbar);
    diff_U = U_current - U_ref;
    l2_errors(j) = sqrt(sum(diff_U(:).^2) * h^2) / sqrt(sum(U_ref(:).^2) * h^2);
end

%% 5. FIGURE 101: SHOCASING THE MORPHOLOGY EVOLUTION (IC SMOOTH vs FINAL TF)
fprintf('\nStep 4: Generating paper-ready morphology showcase panel...\n');
figure(101); clf;
set(gcf, 'Position', [150 150 1100 500], 'Color', 'w');

% Left Plot: Smooth IC at t = 1.0
subplot(1,2,1);
pcolor(X, Y, U_start); shading interp; colormap(parula); colorbar;
ax = gca; ax.FontSize = 16; ax.TickLabelInterpreter = 'latex';
axis equal tight;
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 18);
title('Smooth State $U_{\mathrm{start}}$ at $t = 1.0$', 'Interpreter', 'latex', 'FontSize', 18);

% Right Plot: True Reference solution at total time (t_smooth + t_duration = 2.0)
subplot(1,2,2);
pcolor(X, Y, U_ref); shading interp; colormap(parula); colorbar;
ax = gca; ax.FontSize = 16; ax.TickLabelInterpreter = 'latex';
axis equal tight;
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 18);
title('Reference Solution $U_{\mathrm{ref}}$ at $t = 2.0$', 'Interpreter', 'latex', 'FontSize', 18);

sgtitle('BCP 2D Phase-Field Snapshots for Temporal Refinement Study', 'Interpreter', 'latex', 'FontSize', 18);

%% 6. FIGURE 100: LOG-LOG ERROR PLOT
fprintf('Step 5: Generating convergence plot...\n');
figure(100); clf;
set(gcf, 'Color', 'w');
loglog(dt_array, l2_errors, 'ro-', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'BHM Scheme ($L^2$ Error)');
hold on; grid on; grid minor;

ideal_slope = dt_array .* (l2_errors(end) / dt_array(end));
loglog(dt_array, ideal_slope, 'k--', 'LineWidth', 2, 'DisplayName', 'Theoretical $\mathcal{O}(\Delta t)$');

ax = gca; ax.FontSize = 18; ax.TickLabelInterpreter = 'latex';
xlabel('Time Step Size $\Delta t$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Relative $L^2$ Error', 'Interpreter', 'latex', 'FontSize', 18);
title('Temporal Convergence Study', 'Interpreter', 'latex', 'FontSize', 18);
legend('Interpreter', 'latex', 'FontSize', 14, 'Location', 'southeast');
axis tight;

total_time = toc;
fprintf('\nDone! Showcase plots completed in %.2f minutes.\n', total_time/60);

%% =========================================================================
%% HELPER FUNCTIONS (ALL GPU RESIDENT)
%% =========================================================================
function [U, TheUbar] = BCP2D_pre_evolve(dt, M1, tfinal, N, epsilon, alpha)
    M_dom = 2; a = 0; b = M_dom*pi; h = (b-a)/N;
    k = gpuArray([[0:N/2] [-N/2+1:-1]] ./ (M_dom/2));
    [kx, ky] = meshgrid(k.^2, k.^2); k2 = kx + ky; k4 = k2.^2;
    rng(1527, 'twister');
    U = gpuArray(0.01*rand(N, N) + 0.35); 
    TheUbar = (1/(2*pi))^2 * h^2 * sum(U(:));
    eps2 = epsilon^2;
    lhs = 1 + dt*M1*k4*eps2 + dt*alpha;
    hat_U = fft2(U); t = 0;
    while t < tfinal - dt/2
        RHS = eps2*(M1-1)*ifft2(k4.*hat_U) + ifft2(-k2.*fft2(U.^3 - U)) + alpha*TheUbar;
        hat_U = (hat_U + dt.*fft2(RHS)) ./ lhs;
        U = real(ifft2(hat_U));
        t = t + dt;
    end
    U = gather(U);
end

function U = BCP2D_run_segment(U_start, dt, M1, t_duration, N, epsilon, alpha, TheUbar)
    M_dom = 2; a = 0; b = M_dom*pi; h = (b-a)/N;
    k = gpuArray([[0:N/2] [-N/2+1:-1]] ./ (M_dom/2));
    [kx, ky] = meshgrid(k.^2, k.^2); k2 = kx + ky; k4 = k2.^2;
    U = gpuArray(U_start);
    eps2 = epsilon^2;
    lhs = 1 + dt*M1*k4*eps2 + dt*alpha;
    hat_U = fft2(U); t = 0;
    while t < t_duration - dt/2
        RHS = eps2*(M1-1)*ifft2(k4.*hat_U) + ifft2(-k2.*fft2(U.^3 - U)) + alpha*TheUbar;
        hat_U = (hat_U + dt.*fft2(RHS)) ./ lhs;
        U = real(ifft2(hat_U));
        t = t + dt;
    end
    U = gather(U);
end

function tf = x_exist(name, type)
    tf = exist(name, type) == 2;
end
