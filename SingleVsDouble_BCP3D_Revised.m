%% SingleVsDouble_BCP3D_Revised.m
%% ENGwCOM - Rev 1  --  Section "Single versus Double Precision"
%
%  FULL double run vs FULL single run.
%  Everything in the single run is single: operators, wavenumbers, source
%  term, FFTs, nonlinear term, AND the mass measurement.
%  Everything in the double run is double.
%  Only the initial data is shared: U0 is created once and cast to single
%  (as requested by Reviewer 2), so both runs start from matched data.
%
%  N = 256^3, Omega = [0,2*pi]^3, M1 = 1, eps = 0.1, dt = 0.01, tf = 100.
%
%  Outputs:
%    Figure 400: relative mass drift |M(t)-M(0)|/M(0), each in its own precision
%    Figure 401: relative L2 difference ||U_single - U_double|| / ||U_double||
%  Then two short separate runs measure time per step and peak memory.
clear; close all;

%% 1. PARAMETERS
N        = 128;
M1       = 1;
epsilon  = 0.1;
alpha    = 15;
ubar     = 0.05;     % must be nonzero: M(0) divides the drift
dt       = 0.01;
tf       = 100;
nsteps   = round(tf/dt);
diff_every   = 10;   % compute the field difference every 10 steps
timing_steps = 500;  % steps for the separate timing/memory runs

%% 2. DEVICE BANNER
g = gpuDevice; reset(g); g = gpuDevice;
fprintf('\n=============================================================\n');
fprintf('  FULL SINGLE vs FULL DOUBLE on GPU:  %s\n', g.Name);
fprintf('  N = %d^3, %d steps, alpha = %g, ubar = %g\n', N, nsteps, alpha, ubar);
fprintf('=============================================================\n\n');

%% 3. INITIAL DATA (created once, then cast)
h  = 2*pi/N; x = h*(0:N-1);
[X, Y, Z] = ndgrid(x, x, x);
U0 = ubar + 0.05*( cos(X).*cos(2*Y).*cos(3*Z) ...
                 + cos(3*X).*cos(Y).*cos(2*Z) ...
                 + cos(2*X).*cos(3*Y).*cos(Z) );
clear X Y Z

%% 4. OPERATORS, each built entirely in its own precision
ops_d = build_ops(N, M1, epsilon, alpha, dt, 'double');
ops_s = build_ops(N, M1, epsilon, alpha, dt, 'single');

Ud = gpuArray(U0);              % double
Us = gpuArray(single(U0));      % single
clear U0

% Mass and source term: double in double, single in single
M0d   = mean(Ud(:));
M0s   = mean(Us(:));
src_d = ops_d.dt*N^3*ops_d.alpha*M0d/(1 + ops_d.dt*ops_d.alpha);
src_s = ops_s.dt*single(N)^3*ops_s.alpha*M0s/(1 + ops_s.dt*ops_s.alpha);

Hd = fftn(Ud); Hs = fftn(Us);

drift_d = zeros(1, nsteps);            % stored as double just for plotting
drift_s = zeros(1, nsteps);
tdiff   = (diff_every:diff_every:nsteps)*dt;
fdiff   = zeros(1, numel(tdiff)); id = 0;

%% 5. RUN (both advance in the same loop, each in its own precision)
fprintf('Running full double and full single (%d steps)...\n', nsteps);
tic;
for it = 1:nsteps
    % ---- FULL DOUBLE ----
    Hd = ops_d.A1.*Hd - ops_d.A2.*fftn(Ud.*(Ud.^2 - 1)); Hd(1,1,1) = Hd(1,1,1) + src_d;
    Ud = real(ifftn(Hd));
    % ---- FULL SINGLE ----
    Hs = ops_s.A1.*Hs - ops_s.A2.*fftn(Us.*(Us.^2 - 1)); Hs(1,1,1) = Hs(1,1,1) + src_s;
    Us = real(ifftn(Hs));

    % mass, each measured in its own precision
    drift_d(it) = gather(abs(mean(Ud(:)) - M0d)/abs(M0d));
    drift_s(it) = gather(abs(mean(Us(:)) - M0s)/abs(M0s));

    % field difference (comparison only; does not affect either run)
    if mod(it, diff_every) == 0
        id = id + 1;
        fdiff(id) = gather(norm(double(Us(:)) - Ud(:)) / norm(Ud(:)));
    end
end
wait(g); fprintf('  done in %.1f min\n', toc/60);

t = (1:nsteps)*dt;
fprintf('\nRelative mass drift at tf:  double = %.2e,  single = %.2e\n', drift_d(end), drift_s(end));
fprintf('Max relative mass drift:    double = %.2e,  single = %.2e\n', max(drift_d), max(drift_s));
fprintf('Relative L2 difference at tf (single vs double) = %.2e\n', fdiff(end));
clear Ud Us Hd Hs ops_d ops_s

%% 6. SEPARATE TIMING / MEMORY RUNS (one precision at a time)
fprintf('\nTiming runs (%d steps each)...\n', timing_steps);
[tps_d, peak_d] = timing_run(N, M1, epsilon, alpha, dt, ubar, timing_steps, 'double');
[tps_s, peak_s] = timing_run(N, M1, epsilon, alpha, dt, ubar, timing_steps, 'single');
fprintf('  double: %.4f s/step, peak %.2f GiB\n', tps_d, peak_d);
fprintf('  single: %.4f s/step, peak %.2f GiB\n', tps_s, peak_s);
fprintf('  single/double: time %.2f, memory %.2f\n', tps_s/tps_d, peak_s/peak_d);

save('SingleVsDouble_BCP3D_Revised.mat', 't', 'drift_d', 'drift_s', 'tdiff', 'fdiff', ...
     'tps_d', 'tps_s', 'peak_d', 'peak_s', 'N', 'alpha', 'ubar', 'epsilon', 'M1', 'dt', 'tf');

%% 7. FIGURES
if ~exist('SinglevsDouble', 'dir'), mkdir('SinglevsDouble'); end
pd = drift_d; pd(pd == 0) = NaN;   % exact zeros cannot be shown on a log axis
ps = drift_s; ps(ps == 0) = NaN;

figure(400); clf; set(gcf, 'Color', 'w');
semilogy(t, pd, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Double precision'); hold on;
semilogy(t, ps, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Single precision');
grid on; grid minor;
ax = gca; ax.FontSize = 18; ax.TickLabelInterpreter = 'latex';
xlabel('Time $t$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$|M(t)-M(0)|/M(0)$', 'Interpreter', 'latex', 'FontSize', 18);
title('Relative Mass Drift', 'Interpreter', 'latex', 'FontSize', 18);
legend('Interpreter', 'latex', 'FontSize', 14, 'Location', 'best');
axis tight;
exportgraphics(gcf, 'SinglevsDouble/mass_drift_revised.png', 'Resolution', 300);

figure(401); clf; set(gcf, 'Color', 'w');
semilogy(tdiff, fdiff, 'k-', 'LineWidth', 2.5);
grid on; grid minor;
ax = gca; ax.FontSize = 18; ax.TickLabelInterpreter = 'latex';
xlabel('Time $t$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|U_{\mathrm{single}}-U_{\mathrm{double}}\|_{L^2}/\|U_{\mathrm{double}}\|_{L^2}$', ...
       'Interpreter', 'latex', 'FontSize', 16);
title('Single vs Double: Field Difference', 'Interpreter', 'latex', 'FontSize', 18);
axis tight;
exportgraphics(gcf, 'SinglevsDouble/field_difference_revised.png', 'Resolution', 300);

fprintf('\nSaved SinglevsDouble/mass_drift_revised.png and SinglevsDouble/field_difference_revised.png\n');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================
function ops = build_ops(N, M1, epsilon, alpha, dt, cls)
    % Everything built in precision 'cls' from the start
    k    = gpuArray(cast([0:N/2, -N/2+1:-1], cls));
    dt   = cast(dt, cls);  alpha = cast(alpha, cls);  M1 = cast(M1, cls);
    eps2 = cast(epsilon, cls)^2;
    [KX, KY, KZ] = ndgrid(k, k, k);
    k2 = KX.^2 + KY.^2 + KZ.^2; clear KX KY KZ
    lhs    = 1 + dt*(M1*eps2*k2.^2 + alpha);
    ops.A1 = (1 + dt*(M1-1)*eps2*k2.^2) ./ lhs;
    ops.A2 = dt*k2 ./ lhs;
    ops.dt = dt;  ops.alpha = alpha;
end

function m = smi_used_GiB()
    [~, out] = system('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits');
    m = str2double(strtok(strtrim(out), newline)) / 1024;   % MiB -> GiB
end

function [tps, peak] = timing_run(N, M1, epsilon, alpha, dt, ubar, nsteps, cls)
    g = gpuDevice; reset(g);
    h = 2*pi/N; x = h*(0:N-1);
    [X, Y, Z] = ndgrid(x, x, x);
    U0 = ubar + 0.05*( cos(X).*cos(2*Y).*cos(3*Z) + cos(3*X).*cos(Y).*cos(2*Z) ...
                     + cos(2*X).*cos(3*Y).*cos(Z) );
    clear X Y Z
    ops = build_ops(N, M1, epsilon, alpha, dt, cls);
    U = gpuArray(cast(U0, cls)); clear U0
    src = ops.dt*cast(N, cls)^3*ops.alpha*mean(U(:))/(1 + ops.dt*ops.alpha);
    H = fftn(U);
    peak = smi_used_GiB(); tpoll = 0;
    wait(g); t0 = tic;
    for it = 1:nsteps
        H = ops.A1.*H - ops.A2.*fftn(U.*(U.^2 - 1)); H(1,1,1) = H(1,1,1) + src;
        U = real(ifftn(H));
        if it <= 3 || mod(it, 50) == 0
            tp = tic; wait(g); peak = max(peak, smi_used_GiB()); tpoll = tpoll + toc(tp);
        end
    end
    wait(g); tps = (toc(t0) - tpoll)/nsteps;
end
