%% BCP 3D - Cost Comparison: BHM vs CS vs SAV (wall time + peak GPU memory)
%% ENGwCOM - Rev 1
%  Same straightforward ("naive") implementations as the 2D comparison,
%  extended to 3D with fftn/ifftn. No plots: only timing and memory.
%  For each N and each scheme:
%    - reset the GPU, record baseline memory,
%    - run nsteps time steps,
%    - track peak memory by polling gpuDevice during the run,
%    - record total wall time (synchronized with wait(gpuDevice)).
%  Prints a summary table and LaTeX rows ready to paste into Table tab:scheme_cost.
clear; close all;

%% 1. PARAMETERS
N_list  = [256];
M1      = 1;        % BHM
a_cs    = 2;        % CS
C0      = 1;        % SAV
epsilon = 0.1;
alpha   = 15;
dt      = 0.01;
tf      = 10;       % 1000 steps
nsteps  = round(tf/dt);
poll_every = 10;    % memory polling interval (steps)

schemes = {'BHM', 'CS', 'SAV'};
wall    = zeros(numel(N_list), 3);
peakGB  = zeros(numel(N_list), 3);

%% 2. WARM-UP (small grid, all schemes) so first timings are not penalized
fprintf('Warm-up on N = 64^3 ...\n');
for s = 1:3
    run_scheme(s, 64, 20, M1, a_cs, C0, epsilon, alpha, dt, poll_every);
end

%% 3. MAIN LOOP
for i = 1:numel(N_list)
    N = N_list(i);
    for s = 1:3
        fprintf('\nN = %d^3, %s, %d steps ...\n', N, schemes{s}, nsteps);
        [wall(i,s), peakGB(i,s)] = run_scheme(s, N, nsteps, M1, a_cs, C0, ...
                                              epsilon, alpha, dt, poll_every);
        fprintf('  wall time = %.2f s,  peak memory = %.2f GB\n', wall(i,s), peakGB(i,s));
    end
end

%% 4. SUMMARY
fprintf('\n=== Summary (dt = %g, tf = %g, %d steps, M1 = %g, a = %g) ===\n', ...
        dt, tf, nsteps, M1, a_cs);
fprintf('   N    |  BHM: time   mem   |  CS: time    mem   |  SAV: time   mem   | SAV/BHM time  mem\n');
for i = 1:numel(N_list)
    fprintf('%4d^3  | %8.2f s %5.2f GB | %8.2f s %5.2f GB | %8.2f s %5.2f GB |   %5.2fx   %5.2fx\n', ...
        N_list(i), wall(i,1), peakGB(i,1), wall(i,2), peakGB(i,2), ...
        wall(i,3), peakGB(i,3), wall(i,3)/wall(i,1), peakGB(i,3)/peakGB(i,1));
end

fprintf('\nLaTeX rows:\n');
for i = 1:numel(N_list)
    fprintf('$%d^3$ & %.1f & %.2f & %.1f & %.2f & %.1f & %.2f \\\\\n', N_list(i), ...
        wall(i,1), peakGB(i,1), wall(i,2), peakGB(i,2), wall(i,3), peakGB(i,3));
end

save('BCP3D_Scheme_Cost.mat', 'N_list', 'schemes', 'wall', 'peakGB', ...
     'dt', 'tf', 'M1', 'a_cs', 'C0', 'alpha', 'epsilon');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================
function [twall, peakGB] = run_scheme(s, N, nsteps, M1, a_cs, C0, epsilon, alpha, dt, poll_every)
    g = gpuDevice; reset(g); g = gpuDevice;
    base = g.TotalMemory - g.AvailableMemory;      % memory in use before allocating
    peak = base;

    % Same random initial condition for all schemes
    rng(1527, 'twister');
    U = gpuArray(0.01*rand(N, N, N) + 0.35);
    Ubar = mean(U(:));
    h = 2*pi/N;

    % Wavenumbers (domain [0,2*pi]^3)
    k = gpuArray([0:N/2, -N/2+1:-1]);
    [KX, KY, KZ] = ndgrid(k, k, k);
    k2 = KX.^2 + KY.^2 + KZ.^2; clear KX KY KZ
    k4 = k2.^2;
    eps2 = epsilon^2;

    % Fourier transform of the constant alpha*Ubar (unnormalized fftn)
    S = gpuArray.zeros(N, N, N); S(1,1,1) = N^3*alpha*Ubar;

    wait(g); t0 = tic;
    switch s
        case 1  % BHM
            lhs = 1 + dt*(M1*eps2*k4 + alpha);
            hat_U = fftn(U);
            for it = 1:nsteps
                hat_U = (hat_U + dt*((M1-1)*eps2*k4.*hat_U - k2.*fftn(U.^3 - U) + S)) ./ lhs;
                U = real(ifftn(hat_U));
                if mod(it, poll_every) == 0 || it <= 3, peak = max(peak, g.TotalMemory - g.AvailableMemory); end
            end
        case 2  % CS
            lhs = 1 + dt*(eps2*k4 + a_cs*k2 + alpha);
            hat_U = fftn(U);
            for it = 1:nsteps
                hat_U = (hat_U + dt*(-k2.*fftn(U.^3 - (1 + a_cs)*U) + S)) ./ lhs;
                U = real(ifftn(hat_U));
                if mod(it, poll_every) == 0 || it <= 3, peak = max(peak, g.TotalMemory - g.AvailableMemory); end
            end
        case 3  % SAV (standard first order, r^{n+1} implicit)
            A = 1/dt + eps2*k4 + alpha;
            hat_U = fftn(U);
            E1 = @(V) h^3*sum(0.25*(V(:).^2 - 1).^2);
            r  = sqrt(E1(U) + C0);
            for it = 1:nsteps
                bvec  = (U.^3 - U) / sqrt(E1(U) + C0);
                p_hat = (hat_U/dt + S) ./ A;
                q_hat = -k2.*fftn(bvec) ./ A;
                p     = real(ifftn(p_hat));
                q     = real(ifftn(q_hat));
                r     = (r + 0.5*h^3*sum(bvec(:).*(p(:) - U(:)))) / ...
                        (1 - 0.5*h^3*sum(bvec(:).*q(:)));
                U     = p + r*q;
                hat_U = p_hat + r*q_hat;
                if mod(it, poll_every) == 0 || it <= 3, peak = max(peak, g.TotalMemory - g.AvailableMemory); end
            end
    end
    wait(g); twall = toc(t0);

    % Sanity check: mass conservation
    fprintf('  mass drift = %.2e\n', gather(abs(mean(U(:)) - Ubar)));

    peakGB = (peak - base)/1e9;   % memory used by the run (excludes CUDA context)
    clear U hat_U k2 k4 S lhs A p q bvec p_hat q_hat
end
