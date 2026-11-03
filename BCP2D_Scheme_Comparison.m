%% BCP 2D - Temporal Error Comparison: BHM vs CS vs SAV (Classic Scaling)
%% ENGwCOM - Rev 1
%  Same parameters and smooth initial condition as the convergence study.
%  All three schemes are compared against ONE common reference solution,
%  obtained by Richardson extrapolation of BHM runs with dt_ref and dt_ref/2
%  (second-order accurate, so the reference error is negligible).
%
%  BHM : M1 implicit biharmonic stabilization (linear terms in Fourier space)
%  CS  : linear convex splitting with parameter a (implicit eps^2*Lap^2 - a*Lap)
%  SAV : standard first-order SAV (Shen, Xu, Yang 2018), r^{n+1} implicit,
%        U^{n+1} = p + r^{n+1} q, scalar equation for r^{n+1}
tic;
clear; close all;

%% 1. GENERAL PARAMETERS (same as convergence study)
N       = 256;
M1      = 1;       % BHM
a_cs    = 2;       % CS splitting parameter (classical value)
C0      = 1;       % SAV constant
epsilon = 0.1;
alpha   = 0.1;

t_smooth   = 1.4;
t_duration = 1.00;

dt_array = 0.025 ./ (2.^(0:8));

% --- CONTROL FLAG FOR REFERENCE SOLUTION ---
compute_ref = false;   % Set to FALSE after the first run to load from disk
% --------------------------------------------
ref_file = 'BCP_Comparison_Reference_Data.mat';
dt_ref   = 1e-5;

%% 2. SMOOTH INITIAL CONDITION (identical to convergence study)
fprintf('Step 1: Evolving random noise to t = %.1f to obtain smooth state...\n', t_smooth);
[U_start, TheUbar] = BCP2D_pre_evolve(0.01, M1, t_smooth, N, epsilon, alpha);

%% 3. COMMON REFERENCE SOLUTION (Richardson extrapolation of BHM)
if compute_ref
    fprintf('\nStep 2: Reference solution (Richardson, dt = %g and %g)...\n', dt_ref, dt_ref/2);
    U1 = run_BHM(U_start, dt_ref,   M1, t_duration, N, epsilon, alpha, TheUbar);
    U2 = run_BHM(U_start, dt_ref/2, M1, t_duration, N, epsilon, alpha, TheUbar);
    U_ref = 2*U2 - U1;
    fprintf('        -> Richardson correction size: %.2e (relative)\n', ...
            norm(U2(:)-U1(:))/norm(U_ref(:)));
    save(ref_file, 'U_ref', 'dt_ref');
    fprintf('        -> Saved to %s\n', ref_file);
else
    fprintf('\nStep 2: Loading reference solution from disk...\n');
    if x_exist(ref_file, 'file')
        load(ref_file, 'U_ref');
    else
        error('%s not found! Please set compute_ref = true for this run.', ref_file);
    end
end

%% 4. ERROR SWEEP FOR THE THREE SCHEMES
schemes = {'BHM', 'CS', 'SAV'};
errs    = zeros(3, length(dt_array));
wall    = zeros(1, 3);
M_dom = 2; a = 0; b = M_dom*pi; h = (b-a)/N;

for s = 1:3
    fprintf('\nStep 3.%d: %s sweep over %d time steps...\n', s, schemes{s}, length(dt_array));
    t0 = tic;
    for j = 1:length(dt_array)
        dt = dt_array(j);
        switch s
            case 1, U = run_BHM(U_start, dt, M1,   t_duration, N, epsilon, alpha, TheUbar);
            case 2, U = run_CS (U_start, dt, a_cs, t_duration, N, epsilon, alpha, TheUbar);
            case 3, U = run_SAV(U_start, dt, C0,   t_duration, N, epsilon, alpha, TheUbar);
        end
        d = U - U_ref;
        errs(s,j) = sqrt(sum(d(:).^2)*h^2) / sqrt(sum(U_ref(:).^2)*h^2);
    end
    wall(s) = toc(t0);
    rates = log2(errs(s,1:end-1)./errs(s,2:end));
    fprintf('  errors: %s\n', sprintf('%9.2e', errs(s,:)));
    fprintf('  rates : %s\n', sprintf('%9.2f', rates));
    fprintf('  wall time for sweep: %.2f s\n', wall(s));
end

fprintf('\n=== Summary ===\n');
fprintf('Scheme   err(dt=%.1e)   err(dt=%.1e)   ratio to BHM (smallest dt)   sweep wall time\n', ...
        dt_array(1), dt_array(end));
for s = 1:3
    fprintf('%-6s   %.3e        %.3e        %6.2f                       %.2f s\n', ...
        schemes{s}, errs(s,1), errs(s,end), errs(s,end)/errs(1,end), wall(s));
end

save('BCP_Comparison_Errors.mat', 'dt_array', 'errs', 'wall', 'schemes', 'M1', 'a_cs', 'C0');

%% 5. FIGURE 300: ERROR COMPARISON (same style as temporal plot)
figure(300); clf;
set(gcf, 'Color', 'w');
loglog(dt_array, errs(1,:), 'ro-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
       'MarkerFaceColor', 'r', 'DisplayName', sprintf('BHM ($M_1=%g$)', M1)); hold on;
loglog(dt_array, errs(2,:), 'bs-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
       'MarkerFaceColor', 'b', 'DisplayName', sprintf('CS ($a=%g$)', a_cs));
loglog(dt_array, errs(3,:), 'd--', 'Color', [0 0.6 0], 'LineWidth', 2.5, 'MarkerSize', 12, ...
       'MarkerFaceColor', 'none', 'DisplayName', 'SAV');
grid on; grid minor;

ideal_slope = dt_array .* (min(errs(:,end)) / dt_array(end)) / 2;
loglog(dt_array, ideal_slope, 'k--', 'LineWidth', 2, 'DisplayName', '$\mathcal{O}(\Delta t)$');

ax = gca; ax.FontSize = 18; ax.TickLabelInterpreter = 'latex';
xlabel('Time Step Size $\Delta t$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Relative $L^2$ Error', 'Interpreter', 'latex', 'FontSize', 18);
title('Temporal Error: BHM, CS, SAV', 'Interpreter', 'latex', 'FontSize', 18);
legend('Interpreter', 'latex', 'FontSize', 14, 'Location', 'southeast');
axis tight;

total_time = toc;
fprintf('\nDone! Comparison completed in %.2f minutes.\n', total_time/60);

%% =========================================================================
%% HELPER FUNCTIONS (ALL GPU RESIDENT)
%% =========================================================================
function [kx2, k4] = wavenumbers(N)
    M_dom = 2;
    k = gpuArray([[0:N/2] [-N/2+1:-1]] ./ (M_dom/2));
    [kx, ky] = meshgrid(k.^2, k.^2); kx2 = kx + ky; k4 = kx2.^2;
end

function S = mean_source(N, alpha, TheUbar)
    % Fourier transform of the constant alpha*Ubar (unnormalized fft2)
    S = gpuArray.zeros(N, N);
    S(1,1) = N^2 * alpha * TheUbar;
end

function [U, TheUbar] = BCP2D_pre_evolve(dt, M1, tfinal, N, epsilon, alpha)
    % Identical to the convergence-study pre-evolution
    M_dom = 2; a = 0; b = M_dom*pi; h = (b-a)/N;
    [k2, k4] = wavenumbers(N);
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

function U = run_BHM(U0, dt, M1, T, N, epsilon, alpha, TheUbar)
    [k2, k4] = wavenumbers(N); eps2 = epsilon^2;
    S   = mean_source(N, alpha, TheUbar);
    lhs = 1 + dt*(M1*eps2*k4 + alpha);
    U = gpuArray(U0); hat_U = fft2(U);
    for it = 1:round(T/dt)
        hat_U = (hat_U + dt*((M1-1)*eps2*k4.*hat_U - k2.*fft2(U.^3 - U) + S)) ./ lhs;
        U = real(ifft2(hat_U));
    end
    U = gather(U);
end

function U = run_CS(U0, dt, a_cs, T, N, epsilon, alpha, TheUbar)
    [k2, k4] = wavenumbers(N); eps2 = epsilon^2;
    S   = mean_source(N, alpha, TheUbar);
    lhs = 1 + dt*(eps2*k4 + a_cs*k2 + alpha);
    U = gpuArray(U0); hat_U = fft2(U);
    for it = 1:round(T/dt)
        hat_U = (hat_U + dt*(-k2.*fft2(U.^3 - (1 + a_cs)*U) + S)) ./ lhs;
        U = real(ifft2(hat_U));
    end
    U = gather(U);
end

function U = run_SAV(U0, dt, C0, T, N, epsilon, alpha, TheUbar)
    [k2, k4] = wavenumbers(N); eps2 = epsilon^2;
    h = 2*pi/N;
    S = mean_source(N, alpha, TheUbar);
    A = 1/dt + eps2*k4 + alpha;
    U = gpuArray(U0); hat_U = fft2(U);
    E1 = @(V) h^2*sum(0.25*(V(:).^2 - 1).^2);
    r  = sqrt(E1(U) + C0);
    for it = 1:round(T/dt)
        bvec = (U.^3 - U) / sqrt(E1(U) + C0);
        p    = real(ifft2((hat_U/dt + S) ./ A));
        q    = real(ifft2(-k2.*fft2(bvec) ./ A));
        r    = (r + 0.5*h^2*sum(bvec(:).*(p(:) - U(:)))) / ...
               (1 - 0.5*h^2*sum(bvec(:).*q(:)));
        U    = p + r*q;
        hat_U = fft2(U);
    end
    U = gather(U);
end

function tf = x_exist(name, type)
    tf = exist(name, type) == 2;
end
