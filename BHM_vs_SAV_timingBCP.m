%% Timing Comparison: BHM vs SAV for 2D Block Copolymer (BCP), alpha=10
clear; clc; close all;

%% Shared Parameters
N      = 128;
Lx     = 2*pi;
eps    = 0.05;
dt     = 0.01;
tfinal = 100;
h      = Lx/N;
alpha  = 10;

%% Shared initial condition (same seed for both)
rng(42, 'twister');
u0 = 0.1 + 0.01*randn(N, N);

%% Wavenumbers (GPU)
k  = gpuArray([[0:N/2] [-N/2+1:-1]] * (2*pi/Lx));
[KX, KY] = meshgrid(k, k);
K2 = KX.^2 + KY.^2;
K4 = K2.^2;
K2inv = K2; K2inv(1,1) = 1;   % avoid div-by-zero; numerator is 0 there anyway

F  = @(u) 0.25*(u.^2 - 1).^2;
fp = @(u) u.^3 - u;
nsteps = round(tfinal/dt);

%==========================================================
%% BHM METHOD
%==========================================================
fprintf('Running BHM (BCP, alpha=%g)...\n', alpha);
M1   = 5;
u    = gpuArray(u0);
lhs  = 1 + dt*M1*eps^2*K4 + dt*alpha;
hat_u = fft2(u);

Energy_BHM   = zeros(1, nsteps);
time_vec     = zeros(1, nsteps);

tic;
t = 0;
for it = 1:nsteps
    Ubar = mean(u(:));
    RHS   = eps^2*(M1-1)*ifft2(K4.*hat_u) ...
          + ifft2(-K2.*fft2(u.^3 - u)) ...
          + alpha*Ubar;
    hat_u = (hat_u + dt*fft2(RHS)) ./ lhs;
    u     = real(ifft2(hat_u));
    t     = t + dt;

    grad_sq   = real(sum(sum(K2.*abs(hat_u).^2)))/N^4;
    psi_hat   = hat_u ./ K2inv;  psi_hat(1,1) = 0;
    psi_gsq   = real(sum(sum(K2.*abs(psi_hat).^2)))/N^4;

    Energy_BHM(it) = gather(0.5*eps^2*grad_sq + h^2*sum(sum(F(u))) ...
                             + 0.5*alpha*psi_gsq);
    time_vec(it)   = t;
end
u = gather(u);
time_BHM = toc;
fprintf('BHM done: %.2f seconds (%.4f min)\n', time_BHM, time_BHM/60);

%==========================================================
%% SAV METHOD
%==========================================================
fprintf('Running SAV (BCP, alpha=%g)...\n', alpha);
C0   = 1;
u    = gpuArray(u0);
E1_init = gather(h^2 * sum(sum(F(u))));
r    = gpuArray(sqrt(E1_init + C0));
hat_u = fft2(u);

A_diag      = 1/dt + eps^2*K4 + alpha;
Energy_SAV  = zeros(1, nsteps);

tic;
t = 0;
for it = 1:nsteps
    Ubar  = mean(u(:));
    E1_n  = h^2 * sum(sum(F(u)));
    sqE   = sqrt(E1_n + C0);
    xi_n  = r;                          % FIXED: was r / sqE (double division bug)
    b_hat = fft2(fp(u) / sqE);

    rhs_hat      = hat_u/dt - K2.*(xi_n*b_hat) + fft2(alpha*Ubar*ones(N,N));
    hat_u_new    = rhs_hat ./ A_diag;
    u_new        = real(ifft2(hat_u_new));

    dudt = real(ifft2((hat_u_new - hat_u)/dt));
    r    = r + dt*(1/(2*sqE))*(h^2*sum(sum(fp(u).*dudt)));

    u     = u_new;
    hat_u = hat_u_new;
    t     = t + dt;

    grad_sq   = real(sum(sum(K2.*abs(hat_u).^2)))/N^4;
    psi_hat   = hat_u ./ K2inv;  psi_hat(1,1) = 0;
    psi_gsq   = real(sum(sum(K2.*abs(psi_hat).^2)))/N^4;

    Energy_SAV(it) = gather(0.5*eps^2*grad_sq + h^2*sum(sum(F(u))) ...
                             + 0.5*alpha*psi_gsq);
end
u = gather(u);
time_SAV = toc;
fprintf('SAV done: %.2f seconds (%.4f min)\n', time_SAV, time_SAV/60);

%==========================================================
%% Summary
%==========================================================
fprintf('\n=== Timing Summary (BCP, alpha=%g) ===\n', alpha);
fprintf('BHM: %.2f sec (%.4f min)\n', time_BHM, time_BHM/60);
fprintf('SAV: %.2f sec (%.4f min)\n', time_SAV, time_SAV/60);
fprintf('SAV/BHM ratio: %.2fx\n', time_SAV/time_BHM);

%==========================================================
%% Plots
%==========================================================
figure(1); clf;
semilogx(time_vec, Energy_BHM, 'b-', 'LineWidth', 2.5); hold on;
semilogx(time_vec, Energy_SAV, 'r--', 'LineWidth', 2.5);
grid on; grid minor;
legend('BHM', 'SAV', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Time ($t$)', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Free Energy $\mathcal{E}(u)$', 'Interpreter', 'latex', 'FontSize', 16);
title(sprintf('BHM vs SAV Energy (BCP) -- N=%d, $\\alpha$=%g, dt=%.3f, T=%d', ...
    N, alpha, dt, tfinal), 'Interpreter', 'latex', 'FontSize', 14);
axis tight;