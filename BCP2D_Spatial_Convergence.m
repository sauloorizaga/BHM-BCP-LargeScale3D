%% BCP 2D - Spatial Convergence Study (Classic Scaling)
%% ENGwCOM - Rev 1
%  Same scheme, parameters and helper functions as the temporal study.
%  Smooth IC: pre-evolve once on the fine grid Nf, then spectrally truncate
%  to each coarse grid N (same smooth function on every grid).
%  All grids use the same dt, so the differences are purely spatial.
%  Error: coarse solution vs fine reference sampled at the coarse grid points
%  (nested periodic grids, no interpolation needed).
tic;
clear; close all;

%% 1. GENERAL PARAMETERS
Nf       = 1024;                  % Fine (reference) grid
N_array  = [32 64 128 256 512];   % Coarse grids
M1       = 1;
epsilon  = 0.1;
alpha    = 0.1;

t_smooth   = 1.4;    % Pre-evolution time (same as temporal study)
t_duration = 1.00;   % Integration interval
dt_fix     = 1e-4;   % Same dt for ALL grids

% --- CONTROL FLAG FOR REFERENCE SOLUTION ---
compute_ref = true;   % Set to FALSE after the first run to load from disk
% --------------------------------------------
ref_file = 'BCP_Spatial_Reference_Data.mat';

%% 2. SMOOTH IC ON FINE GRID + REFERENCE SOLUTION (compute or load)
if compute_ref
    fprintf('Step 1: Pre-evolving random noise on Nf = %d to t = %.1f...\n', Nf, t_smooth);
    dt_init = 0.01;
    [U_start_f, TheUbar] = BCP2D_pre_evolve(dt_init, M1, t_smooth, Nf, epsilon, alpha);

    fprintf('Step 2: Computing reference solution on Nf = %d with dt = %g...\n', Nf, dt_fix);
    U_ref = BCP2D_run_segment(U_start_f, dt_fix, M1, t_duration, Nf, epsilon, alpha, TheUbar);
    save(ref_file, 'U_start_f', 'U_ref', 'TheUbar', 'Nf', 'dt_fix', 't_smooth', 't_duration');
    fprintf('        -> Saved to %s\n', ref_file);
else
    fprintf('Step 1-2: Loading smooth IC and reference solution from disk...\n');
    if x_exist(ref_file, 'file')
        load(ref_file, 'U_start_f', 'U_ref', 'TheUbar');
    else
        error('%s not found! Please set compute_ref = true for this run.', ref_file);
    end
end

%% 3. SPATIAL REFINEMENT LOOP
fprintf('\nStep 3: Running spatial refinement loop across %d grids...\n', length(N_array));
M_dom = 2; a = 0; b = M_dom*pi;
l2_errors  = zeros(size(N_array));
mass_drift = zeros(size(N_array));
eps_over_h = epsilon ./ ((b-a)./N_array);

for j = 1:length(N_array)
    N = N_array(j);
    h = (b-a)/N;
    fprintf('  -> Grid %d/%d: N = %d (eps/h = %.2f, %d steps)\n', ...
            j, length(N_array), N, eps_over_h(j), round(t_duration/dt_fix));

    U0 = spec_restrict(U_start_f, N);                 % same smooth IC
    U  = BCP2D_run_segment(U0, dt_fix, M1, t_duration, N, epsilon, alpha, TheUbar);

    s     = Nf/N;
    U_r   = U_ref(1:s:end, 1:s:end);                  % reference at coarse points
    diffU = U - U_r;
    l2_errors(j)  = sqrt(sum(diffU(:).^2)*h^2) / sqrt(sum(U_r(:).^2)*h^2);
    mass_drift(j) = abs(mean(U(:)) - TheUbar);
end

fprintf('\n   N      eps/h     Rel. L2 error     Mass drift\n');
for j = 1:length(N_array)
    fprintf('%5d   %6.2f    %.4e       %.2e\n', N_array(j), eps_over_h(j), l2_errors(j), mass_drift(j));
end

%% 4. FIGURE 200: SPATIAL CONVERGENCE PLOT (same style as temporal plot)
fprintf('\nStep 4: Generating spatial convergence plot...\n');
figure(200); clf;
set(gcf, 'Color', 'w');
semilogy(N_array, l2_errors, 'ro-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
         'MarkerFaceColor', 'r', 'DisplayName', 'BHM Scheme ($L^2$ Error)');
hold on; grid on; grid minor;

ax = gca; ax.FontSize = 18; ax.TickLabelInterpreter = 'latex';
ax.XScale = 'log'; ax.XTick = N_array;
xlabel('Grid Size $N$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Relative $L^2$ Error', 'Interpreter', 'latex', 'FontSize', 18);
title('Spatial Convergence Study', 'Interpreter', 'latex', 'FontSize', 18);
legend('Interpreter', 'latex', 'FontSize', 14, 'Location', 'northeast');
axis tight;

total_time = toc;
fprintf('\nDone! Spatial convergence completed in %.2f minutes.\n', total_time/60);

%% =========================================================================
%% HELPER FUNCTIONS (ALL GPU RESIDENT) - same as temporal study
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
    TheUbar = gather(TheUbar);
end

function U = BCP2D_run_segment(U_start, dt, M1, t_duration, N, epsilon, alpha, TheUbar)
    M_dom = 2;
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

function Uc = spec_restrict(Uf, N)
    % Spectral truncation from Nf x Nf to N x N (keeps modes -N/2+1..N/2-1,
    % Nyquist set to zero so the result is real). Preserves the mean exactly.
    Nf  = size(Uf, 1);
    Hf  = fft2(gpuArray(Uf));
    idf = [1:N/2, Nf-N/2+2:Nf];
    idc = [1:N/2, N/2+2:N];
    Hc  = zeros(N, N, 'like', Hf);
    Hc(idc, idc) = Hf(idf, idf);
    Uc  = gather(real(ifft2(Hc)) * (N/Nf)^2);
end

function tf = x_exist(name, type)
    tf = exist(name, type) == 2;
end
