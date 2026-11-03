%% BCP 3D - BHM vs SAV WITH energy monitoring at every step (512^3)
%% ENGwCOM - Rev 1
%  Same straightforward implementations as BCP3D_Scheme_Cost.m, now with the
%  discrete energy E_h (Theorem 4) evaluated at every time step.
%  The energy is computed in Fourier space from hat_U (Parseval), so it needs
%  NO extra FFTs:
%     eps^2/2 ||grad U||^2  = eps^2/2 * h^3/N^3 * sum |k|^2 |hat_U|^2
%     alpha/2 ||grad psi||^2 = alpha/2 * h^3/N^3 * sum_{k~=0} |hat_U|^2 / |k|^2
%     sum_i h^3 F(U_i)       (physical space)
%  Memory: persistent (gpuDevice, between steps) and peak device memory
%  (nvidia-smi, includes temporaries, FFT workspaces, CUDA context).
clear; close all;

%% 1. PARAMETERS
N          = 512;
M1         = 1;      % BHM
C0         = 1;      % SAV
epsilon    = 0.1;
alpha      = 15;
dt         = 0.01;
nsteps     = 200;    % peak memory is reached in the first steps
poll_every = 10;     % memory polling interval (steps)

schemes = {'BHM', 'SAV'};
res = struct();

%% 2. WARM-UP
fprintf('Warm-up on N = 64^3 ...\n');
for s = 1:2
    run_scheme(s, 64, 10, M1, C0, epsilon, alpha, dt, poll_every);
end

%% 3. RUNS
for s = 1:2
    fprintf('\nN = %d^3, %s with energy, %d steps ...\n', N, schemes{s}, nsteps);
    res(s).out = run_scheme(s, N, nsteps, M1, C0, epsilon, alpha, dt, poll_every);
    o = res(s).out;
    fprintf('  wall time        = %.2f s  (%.3f s/step, polling excluded)\n', o.twall, o.twall/nsteps);
    fprintf('  persistent mem   = %.2f GB (gpuDevice)\n', o.persistGB);
    fprintf('  peak device mem  = %.2f GB (nvidia-smi, total)\n', o.peakTotalGB);
    fprintf('  peak minus idle  = %.2f GB (nvidia-smi, run only)\n', o.peakRunGB);
    fprintf('  energy: E(0) = %.6e, E(end) = %.6e, monotone = %d\n', ...
            o.E(1), o.E(end), all(diff(o.E) <= 1e-10*abs(o.E(1))));
    fprintf('  mass drift = %.2e\n', o.mass);
end

%% 4. SUMMARY
fprintf('\n=== Summary: N = %d^3, %d steps, energy every step ===\n', N, nsteps);
fprintf('Scheme | time/step | persistent | peak (total) | peak (run)\n');
for s = 1:2
    o = res(s).out;
    fprintf('%-6s | %7.3f s | %7.2f GB | %9.2f GB | %8.2f GB\n', schemes{s}, ...
        o.twall/nsteps, o.persistGB, o.peakTotalGB, o.peakRunGB);
end
fprintf('SAV/BHM: time %.2fx, peak %.2fx\n', ...
    res(2).out.twall/res(1).out.twall, res(2).out.peakTotalGB/res(1).out.peakTotalGB);

save('BCP3D_BHM_SAV_Energy.mat', 'res', 'schemes', 'N', 'nsteps', 'dt', 'M1', 'C0', 'alpha', 'epsilon');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================
function m = smi_used_bytes()
    [~, out] = system('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits');
    m = str2double(strtrim(out)) * 2^20;   % MiB -> bytes
end

function o = run_scheme(s, N, nsteps, M1, C0, epsilon, alpha, dt, poll_every)
    g = gpuDevice; reset(g); g = gpuDevice;
    idle    = smi_used_bytes();                    % device memory before allocating
    peakSMI = idle;
    base    = g.TotalMemory - g.AvailableMemory;
    peakDev = base;
    tpoll   = 0;

    rng(1527, 'twister');
    U = gpuArray(0.01*rand(N, N, N) + 0.35);
    Ubar = mean(U(:));
    h = 2*pi/N; eps2 = epsilon^2;

    k = gpuArray([0:N/2, -N/2+1:-1]);
    [KX, KY, KZ] = ndgrid(k, k, k);
    k2 = KX.^2 + KY.^2 + KZ.^2; clear KX KY KZ
    k4 = k2.^2;
    k2inv = 1 ./ k2; k2inv(1,1,1) = 0;             % for the nonlocal energy term

    S = gpuArray.zeros(N, N, N); S(1,1,1) = N^3*alpha*Ubar;
    cE = h^3 / N^3;                                % Parseval factor (unnormalized fftn)
    energy = @(U, hU) 0.5*eps2*cE*sum(k2(:).*abs(hU(:)).^2) ...
                    + h^3*sum(0.25*(U(:).^2 - 1).^2) ...
                    + 0.5*alpha*cE*sum(k2inv(:).*abs(hU(:)).^2);
    E = gpuArray.zeros(1, nsteps+1);

    hat_U = fftn(U);
    E(1) = energy(U, hat_U);

    wait(g); t0 = tic;
    switch s
        case 1  % BHM
            lhs = 1 + dt*(M1*eps2*k4 + alpha);
            for it = 1:nsteps
                hat_U = (hat_U + dt*((M1-1)*eps2*k4.*hat_U - k2.*fftn(U.^3 - U) + S)) ./ lhs;
                U = real(ifftn(hat_U));
                E(it+1) = energy(U, hat_U);
                if mod(it, poll_every) == 0 || it <= 3
                    tp = tic; wait(g);
                    peakSMI = max(peakSMI, smi_used_bytes());
                    peakDev = max(peakDev, g.TotalMemory - g.AvailableMemory);
                    tpoll = tpoll + toc(tp);
                end
            end
        case 2  % SAV
            A  = 1/dt + eps2*k4 + alpha;
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
                E(it+1) = energy(U, hat_U);
                if mod(it, poll_every) == 0 || it <= 3
                    tp = tic; wait(g);
                    peakSMI = max(peakSMI, smi_used_bytes());
                    peakDev = max(peakDev, g.TotalMemory - g.AvailableMemory);
                    tpoll = tpoll + toc(tp);
                end
            end
    end
    wait(g); o.twall = toc(t0) - tpoll;

    o.E           = gather(E);
    o.mass        = gather(abs(mean(U(:)) - Ubar));
    o.persistGB   = (peakDev - base)/1e9;
    o.peakTotalGB = peakSMI/1e9;
    o.peakRunGB   = (peakSMI - idle)/1e9;
    clear U hat_U k2 k4 k2inv S lhs A p q bvec p_hat q_hat
end
