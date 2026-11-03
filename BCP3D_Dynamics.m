%% BCP 3D - Computed Dynamics (single file)
%  BHM scheme (nested update) + energy via Parseval (no extra FFTs) on GPU
%  Outputs: 4 isosurface snapshots, energy plot, energy rate plot, .mat file
tic;

%% USER PARAMETERS
N       = 256;
dt      = 0.01;
M1      = 1;
tfinal  = 2000;
epsilon = 0.05;
alpha   = 50;
snap_times = [10, 100, 500, 1000];   ubar = 0.25;   %Gyroid 
%snap_times = [5, 10, 20, 1000];   ubar = 0.37;   %Spheres
snap_times = [350, 400, 1500, 2000];  ubar = 0.0;    %Lamellar (tfinal = 5000)

plot_stride = 2;     % isosurfaces on every 2nd point (512^3 -> 256^3), much faster
tag = ['ubar' num2str(round(ubar*100)) '_N' num2str(N)];

%% RUN SOLVER
[U, time_vec, Energy, snap_U, MassError] = BCP3D_BHM_solver(dt, M1, tfinal, N, epsilon, alpha, ubar, snap_times);

%% SAVE EVERYTHING (in case MATLAB closes overnight)
save(['BCP3D_dynamics_' tag '.mat'], 'U', 'time_vec', 'Energy', 'snap_U', 'snap_times', ...
     'MassError', 'N', 'dt', 'M1', 'tfinal', 'epsilon', 'alpha', 'ubar', '-v7.3');

%% DOMAIN (for plotting)
a = 0; b = 2*pi; h = (b-a)/N;
x = a:h:b-h;
idx = 1:plot_stride:N;
[X,Y,Z] = meshgrid(x(idx), x(idx), x(idx));

%% FIGURES 1-4: Isosurfaces at the snapshot times
for s = 1:4
    Us = snap_U{s}(idx,idx,idx);
    figure(s); clf;
    %isosurface(X,Y,Z,Us,-.15);
    isosurface(X,Y,Z,Us,-.05);
    isosurface(X,Y,Z,Us,.05);
    %isosurface(X,Y,Z,Us,.15);
    ax = gca; ax.FontSize = 14;
    camlight; lighting phong;
    axis([a b a b a b]);
    title(['$t=' num2str(snap_times(s)) '$'],'Interpreter','latex','FontSize',18);
    saveas(gcf, ['BCP3D_snap' num2str(s) '_' tag '.png']);
end

%% FIGURE 5: Energy
figure(5); clf;
semilogx(time_vec(2:end), Energy(2:end), 'b-', 'LineWidth', 2.5);
grid on; grid minor;
ax = gca; ax.FontSize = 14;
ax.TickLabelInterpreter = 'latex';
xlabel('Time ($t$)','Interpreter','latex','FontSize',16);
ylabel('Free Energy $\mathcal{E}_h$','Interpreter','latex','FontSize',16);
title(['BCP 3D Energy -- BHM, $\bar{u}=' num2str(ubar) '$'],'Interpreter','latex','FontSize',18);
axis tight;
saveas(gcf, ['BCP3D_energy_' tag '.png']);

%% FIGURE 6: Energy rate  delta_E = |E^{n+1}-E^n| / (dt |E^n|)
dE = abs(diff(Energy)) ./ (dt*abs(Energy(1:end-1)));
figure(6); clf;
semilogy(time_vec(2:end), dE, 'r-', 'LineWidth', 2.5);
%xline(snap_times, 'k--', 'LineWidth', 1.5);
grid on; grid minor;
ax = gca; ax.FontSize = 14;
ax.TickLabelInterpreter = 'latex';
xlabel('Time ($t$)','Interpreter','latex','FontSize',16);
ylabel('$|\mathcal{E}^{n+1}-\mathcal{E}^n|/(\Delta t\,|\mathcal{E}^n|)$','Interpreter','latex','FontSize',16);
title(['Relative Energy Rate -- $\bar{u}=' num2str(ubar) '$'],'Interpreter','latex','FontSize',18);
axis tight;
saveas(gcf, ['BCP3D_energyrate_' tag '.png']);

%% SUMMARY
fprintf('\nEnergy monotone: %d\n', all(diff(Energy) <= 1e-10*abs(Energy(1))));
fprintf('Energy rate at tfinal: %.2e\n', dE(end));
fprintf('Mass error: %.2e\n', MassError);
a_time = toc;
minutes_hours = [a_time/60, a_time/3600]


%% ========================================================================
function [U, time_vec, Energy, snap_U, MassError] = BCP3D_BHM_solver(dt, M1, tfinal, N, epsilon, alpha, ubar, snap_times)
%BCP3D_BHM_solver  3D BCP equation via BHM scheme (nested) + GPU
%   Energy via Parseval: no extra FFTs, no separate Poisson solve

%% Domain
M_dom = 2;
a = 0; b = M_dom*pi;
h = (b-a)/N;

%% Wavenumbers
k = gpuArray([[0:N/2] [-N/2+1:-1]] ./ (M_dom/2));
[kx, ky, kz] = meshgrid(k.^2, k.^2, k.^2);
k2 = kx + ky + kz;
clear kx ky kz k

%% Initial condition
rng(1527,'twister');
U = gpuArray(0.01*rand(N,N,N) + ubar);
TheUbar = mean(U(:));

%% Nested operators (BHM update folded into A1, A2)
eps2 = epsilon^2;
lhs  = 1 + dt*(M1*eps2*k2.^2 + alpha);
A1   = (1 + dt*(M1-1)*eps2*k2.^2) ./ lhs;
A2   = dt*k2 ./ lhs;
clear lhs
src0 = dt*N^3*alpha*TheUbar/(1 + dt*alpha);     % mean source, k=0 mode only

%% Energy weight (Parseval): gradient + nonlocal (psi_hat = U_hat/|k|^2)
k2(1,1,1) = 1;
W = 0.5*eps2*k2 + 0.5*alpha./k2;  W(1,1,1) = 0;
clear k2
cE = h^3/N^3;

%% Storage
nsteps   = round(tfinal/dt);
time_vec = (0:nsteps)*dt;
Energy   = gpuArray.zeros(1, nsteps+1);
snap_it  = round(snap_times/dt);
snap_U   = cell(length(snap_times), 1);

%% Initial energy
hat_U = fftn(U);
Energy(1) = cE*sum(W(:).*abs(hat_U(:)).^2) + h^3*sum(0.25*(U(:).^2 - 1).^2);

%% Time loop
for it = 1:nsteps
    hat_U = A1.*hat_U - A2.*fftn(U.*(U.^2 - 1));
    hat_U(1,1,1) = hat_U(1,1,1) + src0;
    U = real(ifftn(hat_U));

    Energy(it+1) = cE*sum(W(:).*abs(hat_U(:)).^2) + h^3*sum(0.25*(U(:).^2 - 1).^2);

    % Save snapshots
    s = find(snap_it == it);
    if ~isempty(s)
        snap_U{s} = gather(U);
        fprintf('Snapshot saved at t=%.2f\n', it*dt);
    end
end

%% Gather
Energy = gather(Energy);
MassError = gather(abs(mean(U(:)) - TheUbar));
U = gather(U);
fprintf('\nDone! t=%.1f  MassError=%.2e\n', nsteps*dt, MassError);

end
