%% NewBenchmark_BCP3D.m
%% ENGwCOM - Rev 1  --  NEW CPU/GPU BENCHMARK (Section "CPU and GPU Benchmarks")
%
%  Same code, same problem on every machine:
%    BHM nested update + energy at every step (Parseval), double precision
%    N = 64^3, 128^3, 256^3,  dt = 0.01,  tf = 100 (10,000 steps)
%    M1 = 1, eps = 0.1, alpha = 15, Omega = [0,2*pi]^3, U0 = 0.01*rand + 0.35
%
%  ------------------------------------------------------------------------
%  HOW TO USE
%  STEP 1 (each machine): set MODE = 'run' and the switch USE_GPU, then run.
%      CPU (Intel i9)  : USE_GPU = false
%      Any GPU         : USE_GPU = true   (RTX 2060, RTX 5090 Laptop, MIT GPU)
%    The machine is DETECTED AUTOMATICALLY and printed at the start, and the
%    results are saved as  NewBench_<device>.mat  (e.g. NewBench_CPU.mat,
%    NewBench_NVIDIA_GeForce_RTX_5090_Laptop_GPU.mat).
%    If a machine has more than one GPU, set GPU_INDEX (1, 2, ...).
%
%  STEP 2 (once): put all NewBench_*.mat files in one folder,
%    set MODE = 'table' and run. It prints the timing table, the speedups
%    vs the CPU, a consistency check, and LaTeX rows for Table tab:benchmark.
%  ------------------------------------------------------------------------
clear; close all;

%% 0. SWITCHES
MODE      = 'run';     % 'run'  or  'table'
USE_GPU   = true;      % false = CPU run,  true = GPU run
GPU_INDEX = 1;         % which GPU, if the machine has several

%% 1. PARAMETERS (do not change between machines)
N_list  = [64 128 256];
dt      = 0.01;
tf      = 1000;
nsteps  = round(tf/dt);      % 10,000
M1      = 1;
epsilon = 0.1;
alpha   = 15;

switch MODE
%% =======================================================================
case 'run'
    info = machine_info(USE_GPU, GPU_INDEX);

    % ------------------ BIG BANNER: CHECK THE DEVICE ---------------------
    fprintf('\n=============================================================\n');
    if USE_GPU
        fprintf('  RUNNING ON GPU:  %s\n', info.gpu);
        fprintf('  GPU memory    :  %.1f GB,  driver %s,  compute cap. %s\n', ...
                info.gpuMemGB, info.driver, info.cc);
    else
        fprintf('  RUNNING ON CPU:  %s\n', info.cpu);
    end
    fprintf('  CPU           :  %s (%d threads)\n', info.cpu, info.threads);
    fprintf('  MATLAB        :  %s\n', info.matlab);
    fprintf('  Results file  :  NewBench_%s.mat\n', info.tag);
    fprintf('=============================================================\n\n');
    % ---------------------------------------------------------------------

    % warm-up (not timed)
    run_bhm(32, 20, dt, M1, epsilon, alpha, USE_GPU);

    T = zeros(size(N_list)); Efinal = T; mass = T;
    for i = 1:numel(N_list)
        N = N_list(i);
        fprintf('[%s]  N = %d^3, %d steps ...\n', info.tag, N, nsteps);
        [T(i), Efinal(i), mass(i)] = run_bhm(N, nsteps, dt, M1, epsilon, alpha, USE_GPU);
        fprintf('   wall time = %.2f s (%.2f min, %.4f s/step)\n', T(i), T(i)/60, T(i)/nsteps);
        fprintf('   E(tf) = %.12e,  mass drift = %.1e\n\n', Efinal(i), mass(i));
    end
    save(['NewBench_' info.tag '.mat'], 'info', 'N_list', 'nsteps', 'dt', 'tf', ...
         'M1', 'epsilon', 'alpha', 'T', 'Efinal', 'mass');
    fprintf('Saved NewBench_%s.mat   (device: %s)\n', info.tag, ...
            ternary(USE_GPU, info.gpu, info.cpu));

%% =======================================================================
case 'table'
    files = dir('NewBench_*.mat');
    if isempty(files), error('No NewBench_*.mat files found in this folder.'); end
    names = {files.name};
    % CPU first, then GPUs in alphabetical order
    icpu  = find(strcmp(names, 'NewBench_CPU.mat'));
    if isempty(icpu), error('NewBench_CPU.mat not found: the CPU run is the baseline.'); end
    names = [names(icpu), sort(names(setdiff(1:numel(names), icpu)))];

    labels = {}; Tall = []; Eall = [];
    fprintf('\nMachines found:\n');
    for j = 1:numel(names)
        S = load(names{j});
        if S.info.useGPU, lab = S.info.gpu; else, lab = 'CPU'; end
        labels{end+1} = lab;                         %#ok<AGROW>
        Tall(:, end+1) = S.T(:);                      %#ok<AGROW>
        Eall(:, end+1) = S.Efinal(:);                 %#ok<AGROW>
        fprintf('  %-45s | %s | %s\n', lab, S.info.matlab, ...
                ternary(S.info.useGPU, ['driver ' S.info.driver], S.info.cpu));
    end

    fprintf('\nWall time (min), %d steps:\n', nsteps);
    for j = 1:numel(labels), fprintf('  col %d = %s\n', j, labels{j}); end
    fprintf('   N    '); fprintf('|  col %d   ', 1:numel(labels)); fprintf('\n');
    for i = 1:numel(N_list)
        fprintf('%4d^3  ', N_list(i)); fprintf('| %8.2f ', Tall(i,:)/60); fprintf('\n');
    end

    fprintf('\nSpeedup vs CPU:\n   N    '); fprintf('|  col %d   ', 2:numel(labels)); fprintf('\n');
    for i = 1:numel(N_list)
        fprintf('%4d^3  ', N_list(i)); fprintf('| %7.1fx ', Tall(i,1)./Tall(i,2:end)); fprintf('\n');
    end

    fprintf('\nConsistency check: max relative difference in E(tf) vs CPU = %.1e\n', ...
            max(max(abs(Eall - Eall(:,1)) ./ abs(Eall(:,1)))));

    fprintf('\nLaTeX rows (times in min; columns in the order listed above; speedups vs CPU):\n');
    for i = 1:numel(N_list)
        fprintf('$%d^3$', N_list(i));
        fprintf(' & %.2f', Tall(i,:)/60);
        fprintf(' & %.1f', Tall(i,1)./Tall(i,2:end));
        fprintf(' \\\\\n');
    end
end

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================
function [twall, Ef, mdrift] = run_bhm(N, nsteps, dt, M1, epsilon, alpha, useGPU)
    % Production BHM: nested update, energy at every step (Parseval).
    rng(1527, 'twister');
    U = 0.01*rand(N, N, N) + 0.35;
    k = [0:N/2, -N/2+1:-1];
    if useGPU
        g = gpuDevice; reset(g);
        U = gpuArray(U); k = gpuArray(k);
    end
    Ubar = mean(U(:));
    h = 2*pi/N; eps2 = epsilon^2; cE = h^3/N^3;

    [KX, KY, KZ] = ndgrid(k, k, k);
    k2 = KX.^2 + KY.^2 + KZ.^2; clear KX KY KZ
    A1 = (1 + dt*(M1-1)*eps2*k2.^2) ./ (1 + dt*(M1*eps2*k2.^2 + alpha));
    A2 = dt*k2 ./ (1 + dt*(M1*eps2*k2.^2 + alpha));
    k2(1,1,1) = 1;
    W  = 0.5*eps2*k2 + 0.5*alpha ./ k2; W(1,1,1) = 0;
    clear k2
    src0 = dt*N^3*alpha*Ubar / (1 + dt*alpha);
    energy = @(U, hU) cE*sum(W(:).*abs(hU(:)).^2) + h^3*sum(0.25*(U(:).^2 - 1).^2);
    E = zeros(1, nsteps+1, 'like', U);

    hat_U = fftn(U);
    E(1) = energy(U, hat_U);
    if useGPU, wait(g); end
    t0 = tic;
    for it = 1:nsteps
        hat_U = A1.*hat_U - A2.*fftn(U.*(U.^2 - 1));
        hat_U(1,1,1) = hat_U(1,1,1) + src0;
        U = real(ifftn(hat_U));
        E(it+1) = energy(U, hat_U);
    end
    if useGPU, wait(g); end
    twall = toc(t0);
    Ef = gather(E(end));
    mdrift = gather(abs(mean(U(:)) - Ubar));
end

function info = machine_info(useGPU, gpuIndex)
    info.useGPU  = useGPU;
    info.matlab  = version;
    info.threads = maxNumCompThreads;
    info.cpu     = cpu_name();
    info.gpu = ''; info.driver = ''; info.cc = ''; info.gpuMemGB = 0;
    if useGPU
        g = gpuDevice(gpuIndex);
        info.gpu      = g.Name;
        info.cc       = g.ComputeCapability;
        info.gpuMemGB = g.TotalMemory/1e9;
        info.driver   = num2str(g.DriverVersion);
        [st, s] = system('nvidia-smi --query-gpu=driver_version --format=csv,noheader');
        if st == 0 && ~isempty(strtrim(s))
            s = strsplit(strtrim(s), newline); info.driver = strtrim(s{1});
        end
        info.tag = regexprep(g.Name, '[^A-Za-z0-9]+', '_');
    else
        info.tag = 'CPU';
    end
end

function name = cpu_name()
    name = 'unknown CPU';
    try
        if ispc
            [st, s] = system('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"');
            if st ~= 0 || isempty(strtrim(s))
                [~, s] = system('wmic cpu get name'); s = regexprep(s, 'Name', '');
            end
        elseif ismac
            [~, s] = system('sysctl -n machdep.cpu.brand_string');
        else
            [~, s] = system('lscpu | grep "Model name" | cut -d: -f2');
        end
        s = strtrim(s);
        if ~isempty(s), name = strtrim(strtok(s, newline)); end
    catch
    end
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
