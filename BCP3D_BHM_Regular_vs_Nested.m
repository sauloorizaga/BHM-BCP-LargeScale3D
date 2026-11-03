%% BCP 3D - BHM regular vs BHM nested, both WITH energy at every step (512^3)
%% ENGwCOM - Rev 1
%  REGULAR : same BHM as BCP3D_BHM_SAV_Energy.m (straightforward form).
%            Persistent operator arrays: k2, k4, lhs, k2inv, S   (5 full arrays)
%  NESTED  : same scheme, written to minimize persistent arrays and temporaries:
%            - the update is folded into two precomputed operators
%                  A1 = (1 + dt*(M1-1)*eps^2*k4) ./ lhs,   A2 = dt*k2 ./ lhs
%              so that  hat_U = A1.*hat_U - A2.*fftn(U.*(U.^2 - 1))
%            - nonlinearity in Horner form U.*(U.^2-1)
%            - the mean source acts only on the k = 0 mode (no full array S)
%            - the two Fourier energy terms are folded into one weight
%                  W = 0.5*eps^2*k2 + 0.5*alpha*k2inv
%            Persistent operator arrays: A1, A2, W   (3 full arrays)
%  Both versions produce the same solution up to round-off (checked at the end).
clear; close all;

%% 1. PARAMETERS
N          = 512;
M1         = 1;
epsilon    = 0.1;
alpha      = 15;
dt         = 0.01;
nsteps     = 200;
poll_every = 10;

versions = {'Regular', 'Nested'};
res = struct();

%% 2. WARM-UP
fprintf('Warm-up on N = 64^3 ...\n');
for v = 1:2
    run_bhm(v, 64, 10, M1, epsilon, alpha, dt, poll_every);
end

%% 3. RUNS
for v = 1:2
    fprintf('\nN = %d^3, BHM %s with energy, %d steps ...\n', N, versions{v}, nsteps);
    res(v).out = run_bhm(v, N, nsteps, M1, epsilon, alpha, dt, poll_every);
    o = res(v).out;
    fprintf('  wall time        = %.2f s  (%.3f s/step, polling excluded)\n', o.twall, o.twall/nsteps);
    fprintf('  persistent mem   = %.2f GB (gpuDevice)\n', o.persistGB);
    fprintf('  peak device mem  = %.2f GB (nvidia-smi, total)\n', o.peakTotalGB);
    fprintf('  peak minus idle  = %.2f GB (nvidia-smi, run only)\n', o.peakRunGB);
    fprintf('  energy monotone  = %d,  mass drift = %.2e\n', ...
            all(diff(o.E) <= 1e-10*abs(o.E(1))), o.mass);
end

%% 4. CHECK: same solution
dU = max(abs(res(1).out.Ufinal(:) - res(2).out.Ufinal(:)));
dE = max(abs(res(1).out.E - res(2).out.E)) / abs(res(1).out.E(1));
fprintf('\nCheck: max |U_regular - U_nested| = %.2e,  max rel. energy difference = %.2e\n', dU, dE);

%% 5. SUMMARY
fprintf('\n=== Summary: BHM, N = %d^3, %d steps, energy every step ===\n', N, nsteps);
fprintf('Version  | time/step | persistent | peak (total) | peak (run)\n');
for v = 1:2
    o = res(v).out;
    fprintf('%-8s | %7.3f s | %7.2f GB | %9.2f GB | %8.2f GB\n', versions{v}, ...
        o.twall/nsteps, o.persistGB, o.peakTotalGB, o.peakRunGB);
end
fprintf('Nested saves: %.2f GB peak (%.0f%%), time ratio nested/regular = %.2f\n', ...
    res(1).out.peakTotalGB - res(2).out.peakTotalGB, ...
    100*(1 - res(2).out.peakTotalGB/res(1).out.peakTotalGB), ...
    res(2).out.twall/res(1).out.twall);

for v = 1:2, res(v).out.Ufinal = []; end   % do not save the big arrays
save('BCP3D_BHM_Regular_vs_Nested.mat', 'res', 'versions', 'N', 'nsteps', 'dt', 'M1', 'alpha', 'epsilon');

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================
function m = smi_used_bytes()
    [~, out] = system('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits');
    m = str2double(strtrim(out)) * 2^20;   % MiB -> bytes
end

function o = run_bhm(v, N, nsteps, M1, epsilon, alpha, dt, poll_every)
    g = gpuDevice; reset(g); g = gpuDevice;
    idle    = smi_used_bytes();
    peakSMI = idle;
    base    = g.TotalMemory - g.AvailableMemory;
    peakDev = base;
    tpoll   = 0;

    rng(1527, 'twister');
    U = gpuArray(0.01*rand(N, N, N) + 0.35);
    Ubar = mean(U(:));
    h = 2*pi/N; eps2 = epsilon^2;
    cE = h^3 / N^3;                                % Parseval factor (unnormalized fftn)

    k = gpuArray([0:N/2, -N/2+1:-1]);
    [KX, KY, KZ] = ndgrid(k, k, k);
    k2 = KX.^2 + KY.^2 + KZ.^2; clear KX KY KZ
    E = gpuArray.zeros(1, nsteps+1);

    switch v
        case 1  % ---------------- REGULAR ----------------
            k4    = k2.^2;
            k2inv = 1 ./ k2; k2inv(1,1,1) = 0;
            lhs   = 1 + dt*(M1*eps2*k4 + alpha);
            S     = gpuArray.zeros(N, N, N); S(1,1,1) = N^3*alpha*Ubar;
            energy = @(U, hU) 0.5*eps2*cE*sum(k2(:).*abs(hU(:)).^2) ...
                            + h^3*sum(0.25*(U(:).^2 - 1).^2) ...
                            + 0.5*alpha*cE*sum(k2inv(:).*abs(hU(:)).^2);
            hat_U = fftn(U);
            E(1) = energy(U, hat_U);
            wait(g); t0 = tic;
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

        case 2  % ---------------- NESTED ----------------
            % Fold all operators into A1, A2 (update) and W (energy); free k2.
            A1 = (1 + dt*(M1-1)*eps2*k2.^2) ./ (1 + dt*(M1*eps2*k2.^2 + alpha));
            A2 = dt*k2 ./ (1 + dt*(M1*eps2*k2.^2 + alpha));
            k2(1,1,1) = 1;                         % avoid 0/0; fixed below
            W  = 0.5*eps2*k2 + 0.5*alpha ./ k2;
            W(1,1,1) = 0;                          % k = 0: no gradient, psi_0 = 0
            clear k2
            src0 = dt*N^3*alpha*Ubar / (1 + dt*alpha);   % mean source, k = 0 only
            energy = @(U, hU) cE*sum(W(:).*abs(hU(:)).^2) + h^3*sum(0.25*(U(:).^2 - 1).^2);
            hat_U = fftn(U);
            E(1) = energy(U, hat_U);
            wait(g); t0 = tic;
            for it = 1:nsteps
                hat_U = A1.*hat_U - A2.*fftn(U.*(U.^2 - 1));
                hat_U(1,1,1) = hat_U(1,1,1) + src0;
                U = real(ifftn(hat_U));
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
    o.Ufinal      = gather(U);
    o.persistGB   = (peakDev - base)/1e9;
    o.peakTotalGB = peakSMI/1e9;
    o.peakRunGB   = (peakSMI - idle)/1e9;
end
