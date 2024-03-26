function [TD_Ez, Smatrix_Ez] = FDTD_solver(eps_gama, sigma_gama)

%%% Initialize the EM constants
freq = 800e6;
w = 2 * pi * freq;
eps_o = 8.854187817e-12;
u_o = 4 * pi * 1e-7;
sigma_o = 0;
eps_r_b = 40;
sigma_b = 0.1;
eps_b = eps_r_b * eps_o;
c = 1 / sqrt(u_o * eps_o);

Lx = 512e-3;      % This is the size of the entire FDTD computational region
Ly = 512e-3;      % This is the size of the entire FDTD computational region
dx = 2e-3;
dy = 2e-3;

x_dash = [-250e-3 + dx : dx : 250e-3];
y_dash = [-250e-3 + dx : dy : 250e-3];

[Axis_y, Axis_x] = ndgrid(y_dash, x_dash);

Nx = Ly / dy;            % Number of cells of the entire FDTD region for the magnetic field
Ny = Lx / dx;            % Number of cells of the entire FDTD region for the magnetic field
Nx1 = Ly / dy + 1;       % Number of cells of the entire FDTD region for the electric field (one more cell for the electric field)
Ny1 = Lx / dx + 1;       % Number of cells of the entire FDTD region for the electric field 
dt = dx / (c * sqrt(2));    %%% Choose correct time step (see equations of (4.60a) but it is for 3D case)
FS = 1 / dt;
Nt = 3000;       % Time steps calculated in the FDTD

t_array = [0 : dt : (Nt - 1) * dt];              % Time array


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Nx_brain, Ny_brain] = size(eps_gama);

Bx1 = round((Nx1 - Nx_brain) / 2);            
Bx2 = Nx1 - round((Nx1 - Nx_brain) / 2);

By1 = round((Ny1 - Ny_brain) / 2);
By2 = Ny1 - round((Ny1 - Ny_brain) / 2);

%%% Consider the space filled with matching medium, thus relative permittivity is eps_r_b
eps_z = eps_r_b .* ones(Nx1, Ny1); 
eps_z(Bx1 : Bx2, By1 : By2) = eps_gama;
eps_z = eps_z .* eps_o;

%%% Conductivity in coupling medium is sigma_b
sigma_ez = zeros(Nx1, Ny1) + sigma_b;
sigma_ez(Bx1 : Bx2, By1 : By2) = sigma_gama;
sigma_mx = zeros(Nx1, Ny);
sigma_my = zeros(Nx, Ny1);

%%% ----------------------------------------- Define the Tx and Rx positions ----------------------------------------------- %%%
src_num = 16;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define the Tx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phi_Tx = (0 + pi / 16 : pi / 8 : (2 * pi + pi / 16 - pi / 8));
phi_Tx = circshift(phi_Tx, -4);

Tx_Yradius = 110e-3;
Tx_Xradius = 100e-3;

source_contour_x = 10e-3;
source_contour_y = 5e-3;
Probes_Tx = zeros(src_num, 2);

source_X_Tx = zeros(src_num, 1);
source_Y_Tx = zeros(src_num, 1);

for src = 1 : src_num
    Probes_Tx(src, 1) = source_contour_x - cos(phi_Tx(src)) * Tx_Xradius;
    Probes_Tx(src, 2) = source_contour_y - sin(phi_Tx(src)) * Tx_Yradius;
    
    dis_source_x = abs(Probes_Tx(src, 1) - x_dash);
    [~, source_X_Tx(src)] = min(dis_source_x);
    dis_source_y = abs(Probes_Tx(src, 2) - y_dash);
    [~, source_Y_Tx(src)] = min(dis_source_y);

    % dis_source = sqrt((Probes_Tx(src, 1) - Axis_x(:)) .^ 2 + (Probes_Tx(src, 2) - Axis_y(:)) .^ 2);
    % [~, source_pos_idx] = min(dis_source);
    % [source_Y_Tx(src), source_X_Tx(src)] = ind2sub(size(Axis_x), source_pos_idx);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define the Rx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

src_Rx_N = 16;
phi_Rx_1 = (0 + pi / 16 : pi / 8 : (2 * pi + pi / 16 - pi / 8));
phi_Rx_1 = circshift(phi_Rx_1, -4);

Rx_Yradius_1 = 110e-3;
Rx_Xradius_1 = 100e-3;
Probes_Rx = zeros(src_Rx_N, 2);

source_X_Rx = zeros(src_Rx_N, 1);
source_Y_Rx = zeros(src_Rx_N, 1);

for src_dash = 1 : src_Rx_N
    Probes_Rx(src_dash, 1) = source_contour_x - cos(phi_Rx_1(src_dash)) * Rx_Xradius_1;
    Probes_Rx(src_dash, 2) = source_contour_y - sin(phi_Rx_1(src_dash)) * Rx_Yradius_1;
    
    dis_source_x = abs(Probes_Rx(src_dash, 1) - x_dash);
    [~, source_X_Rx(src_dash)] = min(dis_source_x);
    dis_source_y = abs(Probes_Rx(src_dash, 2) - y_dash);
    [~, source_Y_Rx(src_dash)] = min(dis_source_y);
end

%%% ----------------------------------------------------------------------------------------------------------------------- %%%

%%% Permeability in free space is uo
ux = ones(Nx1, Ny);
ux = ux .* u_o;
uy = ones(Nx, Ny1);
uy = uy .* u_o;

%%% ----------------------------------- Initialize the coefficients in the entire FDTD computational region (without considering the PML) --------------------------------- %%%
%%% Initialize the coefficients in the FDTD for Ez updating
Ceze = (2 * eps_z - dt * sigma_ez) ./ (2 * eps_z + dt * sigma_ez);
Cezhy = (2 * dt) ./ ((2 * eps_z + dt * sigma_ez) * dx);
Cezhx = -1 * (2 * dt) ./ ((2 * eps_z + dt * sigma_ez) * dy);
Cezj = -1 * (2 * dt) ./ (2 * eps_z + dt * sigma_ez);

%%% Initialize the coefficients in the FDTD for Hx updating
Chxh = (2 * ux - dt * sigma_mx) ./ (2 * ux + dt * sigma_mx);
Chxez = -1 * (2 * dt) ./ ((2 * ux + dt * sigma_mx) * dy);
Chxm = -1 * (2 * dt)  ./ (2 * ux + dt * sigma_mx);

%%% Initialize the coefficients in the FDTD for Hy updating
Chyh = (2 * uy - dt * sigma_my) ./ (2 * uy + dt * sigma_my);
Chyez = (2 * dt) ./ ((2 * uy + dt * sigma_my) * dx);
Chym = -1 * (2 * dt) ./ (2 * uy + dt * sigma_my);

%%% ----------------------------------------- Define the PML region ----------------------------------------------- %%%
PML_Ro = 1e-8;     % The reflection coefficient of the finite-width PML medium at normal incidence.
n_PML = 2;         % The order of the PML

n_pml_xn = 10;     % The thickness of PML at xn region is 10 cells
n_pml_xp = 10;     % The thickness of PML at xp region is 10 cells
n_pml_yn = 10;     % The thickness of PML at yn region is 10 cells
n_pml_yp = 10;     % The thickness of PML at yp region is 10 cells

%%% Initialize the conductivity in the PML region
sigma_pex_xn = zeros(n_pml_xn, Ny1 - 2);
sigma_pex_xp = zeros(n_pml_xp, Ny1 - 2);
sigma_pex_yn = zeros(Nx1 - n_pml_xn - n_pml_xp - 2, n_pml_yn);
sigma_pex_yp = zeros(Nx1 - n_pml_xn - n_pml_xp - 2, n_pml_yp);

sigma_pey_xn = zeros(n_pml_xn, Ny1 - n_pml_yp - n_pml_yn - 2);
sigma_pey_xp = zeros(n_pml_xp, Ny1 - n_pml_yp - n_pml_yn - 2);
sigma_pey_yn = zeros(Nx1 - 2, n_pml_yn);
sigma_pey_yp = zeros(Nx1 - 2, n_pml_yp);

sigma_pmx_xn = zeros(n_pml_xn, Ny - 1);
sigma_pmx_xp = zeros(n_pml_xp, Ny - 1);

sigma_pmy_yn = zeros(Nx - 1, n_pml_yn);
sigma_pmy_yp = zeros(Nx - 1, n_pml_yp);

%%% Set up the xn region
sigma_max_xn = -1 * ((n_PML + 1) * eps_b * c * log(PML_Ro)) / (2 * dx * n_pml_xn);
pho_e_xn = [n_pml_xn : -1 : 1] - 3/4;
pho_m_xn = [n_pml_xn : -1 : 1] - 1/4;

for xn_num = 1 : n_pml_xn
	sigma_pex_xn(xn_num, :) = sigma_max_xn * (pho_e_xn(xn_num) / n_pml_xn) ^ n_PML;
	sigma_pmx_xn(xn_num, :) = (u_o / eps_b) * (sigma_max_xn * (pho_m_xn(xn_num) / n_pml_xn) ^ n_PML);
end

%%% Set up the xp region
sigma_max_xp = -1 * ((n_PML + 1) * eps_b * c * log(PML_Ro)) / (2 * dx * n_pml_xp);
pho_e_xp = [1 : 1 : n_pml_xp] - 3/4;
pho_m_xp = [1 : 1 : n_pml_xp] - 1/4;

for xp_num = 1 : n_pml_xp
	sigma_pex_xp(xp_num, :) = sigma_max_xp * (pho_e_xp(xp_num) / n_pml_xp) ^ n_PML;
	sigma_pmx_xp(xp_num, :) = (u_o / eps_b) * (sigma_max_xp * (pho_m_xp(xp_num) / n_pml_xp) ^ n_PML);
end

%%% Set up the yn region
sigma_max_yn = -1 * ((n_PML + 1) * eps_b * c * log(PML_Ro)) / (2 * dy * n_pml_yn);
pho_e_yn = [n_pml_yn : -1 : 1] - 3/4;
pho_m_yn = [n_pml_yn : -1 : 1] - 1/4;

for yn_num = 1 : n_pml_yn
	sigma_pey_yn(:, yn_num) = sigma_max_yn * (pho_e_yn(yn_num) / n_pml_yn) ^ n_PML;
	sigma_pmy_yn(:, yn_num) = (u_o / eps_b) * (sigma_max_yn * (pho_m_yn(yn_num) / n_pml_yn) ^ n_PML);
end

%%% Set up the yp region
sigma_max_yp = -1 * ((n_PML + 1) * eps_b * c * log(PML_Ro)) / (2 * dy * n_pml_yp);
pho_e_yp = [1 : 1 : n_pml_yp] - 3/4;
pho_m_yp = [1 : 1 : n_pml_yp] - 1/4;

for yp_num = 1 : n_pml_yp
	sigma_pey_yp(:, yp_num) = sigma_max_yp * (pho_e_yp(yp_num) / n_pml_yp) ^ n_PML;
	sigma_pmy_yp(:, yp_num) = (u_o / eps_b) * (sigma_max_yp * (pho_m_yp(yp_num) / n_pml_yp) ^ n_PML);
end

%%% --------------------------------------------- Define the FDTD coefficients for the PML ----------------------------------------------- %%%
Cezxe_xn = (2 .* eps_b - dt .* sigma_pex_xn) ./ (2 * eps_b + dt .* sigma_pex_xn);
Cezxe_xp = (2 .* eps_b - dt .* sigma_pex_xp) ./ (2 * eps_b + dt .* sigma_pex_xp);
Cezxe_yn = (2 .* eps_b - dt .* sigma_pex_yn) ./ (2 * eps_b + dt .* sigma_pex_yn);
Cezxe_yp = (2 .* eps_b - dt .* sigma_pex_yp) ./ (2 * eps_b + dt .* sigma_pex_yp);

Cezxhy_xn = (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pex_xn) .* dx);
Cezxhy_xp = (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pex_xp) .* dx);
Cezxhy_yn = (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pex_yn) .* dx);
Cezxhy_yp = (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pex_yp) .* dx);

Cezye_xn = (2 .* eps_b - dt .* sigma_pey_xn) ./ (2 .* eps_b + dt .* sigma_pey_xn);
Cezye_xp = (2 .* eps_b - dt .* sigma_pey_xp) ./ (2 .* eps_b + dt .* sigma_pey_xp);
Cezye_yn = (2 .* eps_b - dt .* sigma_pey_yn) ./ (2 .* eps_b + dt .* sigma_pey_yn);
Cezye_yp = (2 .* eps_b - dt .* sigma_pey_yp) ./ (2 .* eps_b + dt .* sigma_pey_yp);

Cezyhx_xn = -1 .* (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pey_xn) .* dy);
Cezyhx_xp = -1 .* (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pey_xp) .* dy);
Cezyhx_yn = -1 .* (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pey_yn) .* dy);
Cezyhx_yp = -1 .* (2 .* dt) ./ ((2 .* eps_b + dt .* sigma_pey_yp) .* dy);

%%% Only consider the yn and yp regions for Hx component
Chxh_yn = (2 .* u_o - dt .* sigma_pmy_yn) ./ (2 .* u_o + dt .* sigma_pmy_yn);
Chxh_yp = (2 .* u_o - dt .* sigma_pmy_yp) ./ (2 .* u_o + dt .* sigma_pmy_yp);

Chxez_yn = -1 .* (2 .* dt) ./ ((2 .* u_o + dt .* sigma_pmy_yn) .* dy);
Chxez_yp = -1 .* (2 .* dt) ./ ((2 .* u_o + dt .* sigma_pmy_yp) .* dy);

%%% Only consider the xn and xy regions for Hy component
Chyh_xn = (2 .* u_o - dt .* sigma_pmx_xn) ./ (2 .* u_o + dt .* sigma_pmx_xn);
Chyh_xp = (2 .* u_o - dt .* sigma_pmx_xp) ./ (2 .* u_o + dt .* sigma_pmx_xp);

Chyez_xn = (2 .* dt) ./ ((2 .* u_o + dt .* sigma_pmx_xn) .* dx);
Chyez_xp = (2 .* dt) ./ ((2 .* u_o + dt .* sigma_pmx_xp) .* dx);

%%% -------------------------------------- Define the boundary of the intermediate region for different fields ------------------------------------------- %%%
%%% Hy only has intermediate region along x direction
xn_Hy_IM = n_pml_xn + 1;
xp_Hy_IM = Nx - n_pml_xp;
%%% Hx only has intermediate region along y direction
yn_Hx_IM = n_pml_yn + 1;
yp_Hx_IM = Ny - n_pml_yp;
%%% Ez has intermetiate region along x and y directions
yn_Ez_IM = n_pml_yn + 2;
yp_Ez_IM = Ny - n_pml_yp;
xn_Ez_IM = n_pml_xn + 2;
xp_Ez_IM = Nx - n_pml_xp;

% %%% The main loop for the FDTD without adding the PML 
% hwait = waitbar(0,'Please wait...');
% for time_step = 1 : Nt - 1

%%% The main loop for the FDTD including the PML regions
% hwait = waitbar(0,'Please wait...');

%%% ------------------------------------- Initialize the source ------------------------------------------ %%%
%%% The frequency range of the simulation is 200 MHz to 400 MHz, thus the highest frequency is 400 MHz
f_min = 0.5e6;
f_max = 2e9;
lambda_min = c / f_max;    % The minimum wave length
delta_s_max = dx;          % The size of the maximum cell
nc = lambda_min / delta_s_max;      % The number of cells per wavelength
tao = sqrt(2.3) / (pi * f_max);     % Tao decide the width of the Guassian pulse
t_o = sqrt(20) * tao;               % Time shift in order to make the Gaussian pulse has zero value at time = 0
src_num = 16;

freq = 800e6 : 10e6 : 800e6;       % Central frequency is 932.7 MHz
FT_kernal = exp(-1i .* 2 .* pi .* freq .* dt);

trunc_x_low = round(Nx / 8);
trunc_x_up = Nx - round(Nx / 8);
trunc_y_low = round(Ny / 8);
trunc_y_up = Ny - round(Ny / 8);

FD_Ez = zeros(trunc_x_up - trunc_x_low + 1, trunc_y_up - trunc_y_low + 1, length(freq), src_num);
FD_Hx = zeros(trunc_x_up - trunc_x_low + 1, trunc_y_up - trunc_y_low + 1, length(freq), src_num);
FD_Hy = zeros(trunc_x_up - trunc_x_low + 1, trunc_y_up - trunc_y_low + 1, length(freq), src_num);

Smatrix_Ez = zeros(src_num, src_num, length(freq));
Smatrix_Hx = zeros(src_num, src_num, length(freq));
Smatrix_Hy = zeros(src_num, src_num, length(freq));

TD_Ez = zeros(Nt - 1, src_num, src_num);

%%% You can set up a break point at line 286, and then run the code named
%%% "FDTD_solver_interface.m"
% tic
for src = 1 : src_num
%     fprintf('#################################################################\n');
%     fprintf('Processing the %ith source...\n', src);
    
    %%% Initialize the frequency domain matrix
    FT_Ez = zeros(Nx1, Ny1, length(FT_kernal));
    FT_Hx = zeros(Nx1, Ny, length(FT_kernal));
    FT_Hy = zeros(Nx, Ny1, length(FT_kernal));
    
    S_Ez = zeros(src_num, length(FT_kernal));
    S_Hx = zeros(src_num, length(FT_kernal));
    S_Hy = zeros(src_num, length(FT_kernal));
    
    %%% Consider the TMz wave
    Hx = zeros(Nx1, Ny);
    Hy = zeros(Nx, Ny1);
    Ez = zeros(Nx1, Ny1);

    %%% Initialize the E and H field in the PML region (We are dealing with TMz wave, so consider the set up in Figure 7.6)
    Ezx_pml_xn = zeros(Nx1, Ny1);
    Ezx_pml_xp = zeros(Nx1, Ny1);
    Ezx_pml_yn = zeros(Nx1, Ny1);
    Ezx_pml_yp = zeros(Nx1, Ny1);

    Ezy_pml_yn = zeros(Nx1, Ny1);
    Ezy_pml_yp = zeros(Nx1, Ny1);
    Ezy_pml_xn = zeros(Nx1, Ny1);
    Ezy_pml_xp = zeros(Nx1, Ny1);

    % The source position is at (319, 200), (309, 152), (285, 116), (248, 91), (200, 81), (154, 90), (118, 113), (93, 147)-------------------------------------
    Jiz = zeros(Nx, Ny, Nt);
    Jiz(source_Y_Tx(src), source_X_Tx(src),:) =  (-1 * sqrt(2 * exp(1)) / tao) .* (t_array - t_o) .* exp(-1 * (t_array - t_o) .^ 2 ./ tao ^ 2);
    
    for time_step = 1 : Nt - 1
%         if time_step == round(Nt / 4)
%             fprintf('25%% time domain transition calculation has finished...\n');
%         end
%         
%         if time_step == round(Nt / 2)
%             fprintf('50%% time domain transition calculation has finished...\n');
%         end
%         
%         if time_step == round(3 * Nt / 4)
%             fprintf('75%% time domain transition calculation has finished...\n');
%         end
        %%% ------------------------------------- Updating the Hx amd Hy fields in the intermediate region ----------------------------------------
        Hx(2 : Nx, yn_Hx_IM : yp_Hx_IM) = Chxh(2 : Nx, yn_Hx_IM : yp_Hx_IM) .* Hx(2 : Nx, yn_Hx_IM : yp_Hx_IM) + Chxez(2 : Nx, yn_Hx_IM : yp_Hx_IM) ...
                                        .* (Ez(2 : Nx, yn_Hx_IM + 1 : yp_Hx_IM + 1) - Ez(2 : Nx, yn_Hx_IM : yp_Hx_IM));
        Hy(xn_Hy_IM : xp_Hy_IM, 2 : Ny) = Chyh(xn_Hy_IM : xp_Hy_IM, 2 : Ny) .* Hy(xn_Hy_IM : xp_Hy_IM, 2 : Ny) + Chyez(xn_Hy_IM : xp_Hy_IM, 2 : Ny) ...
                                        .* (Ez(xn_Hy_IM + 1 : xp_Hy_IM + 1, 2 : Ny) - Ez(xn_Hy_IM : xp_Hy_IM, 2 : Ny));

        %%% --------------------------------------- Updating the Hx and Hy fields in the PML regions ----------------------------------------------

        %%% Update the Hx in the yn region
        Hx(2 : Nx, 1 : n_pml_yn) = Chxh_yn .* Hx(2 : Nx, 1 : n_pml_yn) + Chxez_yn .* (Ez(2 : Nx, 2 : n_pml_yn + 1) - Ez(2 : Nx, 1 : n_pml_yn));
        %%% Update the Hx in the yp region
        Hx(2 : Nx, yp_Hx_IM + 1 : Ny) = Chxh_yp .* Hx(2 : Nx, yp_Hx_IM + 1 : Ny) + Chxez_yp .* (Ez(2 : Nx, yp_Hx_IM + 2 : Ny1) - Ez(2 : Nx, yp_Hx_IM + 1 : Ny1 - 1));

        %%% Update the Hy in the xn region
        Hy(1 : n_pml_xn, 2 : Ny) = Chyh_xn .* Hy(1 : n_pml_xn, 2 : Ny) + Chyez_xn .* (Ez(2 : n_pml_xn + 1, 2 : Ny) - Ez(1 : n_pml_xn, 2 : Ny));
        %%% Update the Hy in the xp region
        Hy(xp_Hy_IM + 1 : Nx, 2 : Ny) = Chyh_xp .* Hy(xp_Hy_IM + 1 : Nx, 2 : Ny) + Chyez_xp .* (Ez(xp_Hy_IM + 2 : Nx1, 2 : Ny) - Ez(xp_Hy_IM + 1 : Nx1 - 1, 2 : Ny));

        %%% --------------------------------------- Updating the Ez fields in the intermediate rgion ----------------------------------------------
        Ez(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) = Ceze(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) .* Ez(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) ...
        + Cezhy(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) .* (Hy(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) - Hy(xn_Ez_IM - 1 : xp_Ez_IM - 1, yn_Ez_IM : yp_Ez_IM)) ...
        + Cezhx(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) .* (Hx(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) - Hx(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM - 1 : yp_Ez_IM - 1)) ...
        + Cezj(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM) .* Jiz(xn_Ez_IM : xp_Ez_IM, yn_Ez_IM : yp_Ez_IM, time_step);

        %%% --------------------------------------- Updating the Ez fields in the PML rgion ----------------------------------------------

        %%% Update the Ezx and Ezy in the xn region
        Ezx_pml_xn(2 : n_pml_xn + 1, 2 : Ny) = Cezxe_xn .* Ezx_pml_xn(2 : n_pml_xn + 1, 2 : Ny) + Cezxhy_xn .* (Hy(2 : n_pml_xn + 1, 2 : Ny) - Hy(1 : n_pml_xn, 2 : Ny));
        Ezy_pml_xn(2 : n_pml_xn + 1, yn_Ez_IM : yp_Ez_IM) = Cezye_xn .* Ezy_pml_xn(2 : n_pml_xn + 1, yn_Ez_IM : yp_Ez_IM) + Cezyhx_xn .* ...
                                                        (Hx(2 : n_pml_xn + 1, yn_Ez_IM : yp_Ez_IM) - Hx(2 : n_pml_xn + 1, yn_Ez_IM - 1 : yp_Ez_IM - 1));
        %%% Update the Ezx  and Ezy in the xp region
        Ezx_pml_xp(xp_Ez_IM + 1 : Nx, 2 : Ny) = Cezxe_xp .* Ezx_pml_xp(xp_Ez_IM + 1 : Nx, 2 : Ny) + Cezxhy_xp .* (Hy(xp_Ez_IM + 1 : Nx, 2 : Ny) - Hy(xp_Ez_IM : Nx - 1, 2 : Ny));
        Ezy_pml_xp(xp_Ez_IM + 1 : Nx, yn_Ez_IM : yp_Ez_IM) = Cezye_xp .* Ezy_pml_xp(xp_Ez_IM + 1 : Nx, yn_Ez_IM : yp_Ez_IM) + Cezyhx_xp .* ...
                                                            (Hx(xp_Ez_IM + 1 : Nx, yn_Ez_IM : yp_Ez_IM) - Hx(xp_Ez_IM + 1 : Nx, yn_Ez_IM - 1 : yp_Ez_IM - 1));

        %%% Update the Ezx and Ezy in the yn region
        Ezy_pml_yn(2 : Nx, 2 : n_pml_yn + 1) = Cezye_yn .* Ezy_pml_yn(2 : Nx, 2 : n_pml_yn + 1) + Cezyhx_yn .* (Hx(2 : Nx, 2 : n_pml_yn + 1) - Hx(2 : Nx, 1 : n_pml_yn));
        Ezx_pml_yn(xn_Ez_IM : xp_Ez_IM, 2 : n_pml_yn + 1) = Cezxe_yn .* Ezx_pml_yn(xn_Ez_IM : xp_Ez_IM, 2 : n_pml_yn + 1) + Cezxhy_yn .* ...
                                                            (Hy(xn_Ez_IM : xp_Ez_IM, 2 : n_pml_yn + 1) - Hy(xn_Ez_IM - 1 : xp_Ez_IM - 1, 2 : n_pml_yn + 1));

        %%% Update the Ezx and Ezy in the yp region
        Ezy_pml_yp(2 : Nx, yp_Ez_IM + 1 : Ny) = Cezye_yp .* Ezy_pml_yp(2 : Nx, yp_Ez_IM + 1 : Ny) + Cezyhx_yp .* (Hx(2 : Nx, yp_Ez_IM + 1 : Ny) - Hx(2 : Nx, yp_Ez_IM : Ny - 1));
        Ezx_pml_yp(xn_Ez_IM : xp_Ez_IM, yp_Ez_IM + 1 : Ny) = Cezxe_yp .* Ezx_pml_yp(xn_Ez_IM : xp_Ez_IM, yp_Ez_IM + 1 : Ny) + Cezxhy_yp .* ...
                                                            (Hy(xn_Ez_IM : xp_Ez_IM, yp_Ez_IM + 1 : Ny) - Hy(xn_Ez_IM - 1 : xp_Ez_IM - 1, yp_Ez_IM + 1 : Ny));

        %%% ------------------------------------------------ Suprimpose the Ezx and Ezy fields to generate the Ez field in the PML region --------------------------------------------------- %%%
        %%% For yn region without the corner
        Ez(xn_Ez_IM : xp_Ez_IM, 2 : n_pml_yn + 1) = Ezx_pml_yn(xn_Ez_IM : xp_Ez_IM, 2 : n_pml_yn + 1) + Ezy_pml_yn(xn_Ez_IM : xp_Ez_IM, 2 : n_pml_yn + 1);
        %%% For yp region without the corner
        Ez(xn_Ez_IM : xp_Ez_IM, yp_Ez_IM + 1 : Ny) = Ezx_pml_yp(xn_Ez_IM : xp_Ez_IM, yp_Ez_IM + 1 : Ny) + Ezy_pml_yp(xn_Ez_IM : xp_Ez_IM, yp_Ez_IM + 1 : Ny);
        %%% For xn region without the corner
        Ez(2 : n_pml_xn + 1, yn_Ez_IM : yp_Ez_IM) = Ezx_pml_xn(2 : n_pml_xn + 1, yn_Ez_IM : yp_Ez_IM) + Ezy_pml_xn(2 : n_pml_xn + 1, yn_Ez_IM : yp_Ez_IM);
        %%% For xp region without the corner
        Ez(xp_Ez_IM + 1 : Nx, yn_Ez_IM : yp_Ez_IM) = Ezx_pml_xp(xp_Ez_IM + 1 : Nx, yn_Ez_IM : yp_Ez_IM) + Ezy_pml_xp(xp_Ez_IM + 1 : Nx, yn_Ez_IM : yp_Ez_IM);
        %%% For the letf bottom corner (overlap of xn and yn)
        Ez(2 : n_pml_xn + 1, 2 : n_pml_yn + 1) = Ezx_pml_xn(2 : n_pml_xn + 1, 2 : n_pml_yn + 1) + Ezy_pml_yn(2 : n_pml_xn + 1, 2 : n_pml_yn + 1);
        %%% For the right bottom corner (overlap of xp and yn)
        Ez(xp_Ez_IM + 1 : Nx, 2 : n_pml_yn + 1) = Ezx_pml_xp(xp_Ez_IM + 1 : Nx, 2 : n_pml_yn + 1) + Ezy_pml_yn(xp_Ez_IM + 1 : Nx, 2 : n_pml_yn + 1);
        %%% For the top left corner (overlap of xn and yp)
        Ez(2 : n_pml_xn + 1, yp_Ez_IM + 1 : Ny) = Ezx_pml_xn(2 : n_pml_xn + 1, yp_Ez_IM + 1 : Ny) + Ezy_pml_yp(2 : n_pml_xn + 1, yp_Ez_IM + 1 : Ny);
        %%% For the top right corcer (overlap of xp and yp)
        Ez(xp_Ez_IM + 1 : Nx, yp_Ez_IM + 1 : Ny) = Ezx_pml_xp(xp_Ez_IM + 1 : Nx, yp_Ez_IM + 1 : Ny) + Ezy_pml_yp(xp_Ez_IM + 1 : Nx, yp_Ez_IM + 1 : Ny);
        
        for kk = 1 : src_num
            TD_Ez(time_step, src, kk) = Ez(source_Y_Rx(kk), source_X_Rx(kk));
        end
        
        % imagesc(abs(Ez));axis image;
        % pause(0.001)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Implement the discrete Fourier Transform
        for nf = 1 : length(freq)
            FT_Ez(:, :, nf) = FT_Ez(:, :, nf) + (FT_kernal(nf) ^ time_step) * Ez;
            FT_Hx(:, :, nf) = FT_Hx(:, :, nf) + (FT_kernal(nf) ^ time_step) * Hx;
            FT_Hy(:, :, nf) = FT_Hy(:, :, nf) + (FT_kernal(nf) ^ time_step) * Hy;
        end

        for kk = 1 : src_num
            for nf = 1 : length(freq)
                S_Ez(kk, nf) = S_Ez(kk, nf) + (FT_kernal(nf) ^ time_step) * Ez(source_Y_Rx(kk), source_X_Rx(kk));
                S_Hx(kk, nf) = S_Hx(kk, nf) + (FT_kernal(nf) ^ time_step) * Hx(source_Y_Rx(kk), source_X_Rx(kk));
                S_Hy(kk, nf) = S_Hy(kk, nf) + (FT_kernal(nf) ^ time_step) * Hy(source_Y_Rx(kk), source_X_Rx(kk));
            end
        end

        FD_Ez(:, :, :, src) = FT_Ez(trunc_x_low : trunc_x_up, trunc_y_low : trunc_y_up, :);
        FD_Hx(:, :, :, src) = FT_Hx(trunc_x_low : trunc_x_up, trunc_y_low : trunc_y_up, :);
        FD_Hy(:, :, :, src) = FT_Hy(trunc_x_low : trunc_x_up, trunc_y_low : trunc_y_up, :);
        
        Smatrix_Ez(src, :, :) = S_Ez;
        Smatrix_Hx(src, :, :) = S_Hx;
        Smatrix_Hy(src, :, :) = S_Hy;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    end
    
%     clear Jiz Ezx_pml_xn Ezx_pml_xp Ezx_pml_yn Ezx_pml_yp Ezy_pml_yn Ezy_pml_yp Ezy_pml_xn Ezy_pml_xp

end
% toc

% FD_Ez = FD_Ez .* dt;
% FD_Hx = FD_Hx .* dt;
% FD_Hy = FD_Hy .* dt;

Smatrix_Ez = Smatrix_Ez .* dt;
% Smatrix_Hx = Smatrix_Hx .* dt;
% Smatrix_Hy = Smatrix_Hy .* dt;

%%% Verify the FDTD results by using the electric field wave equation

% for ii = 1 : src_num
%     Ez_tot = FD_Ez(:, :, 1, ii);
% 
%     Ez_tot_rec = Ez_tot;
%     [Ez_gy, Ez_gx] = gradient(Ez_tot_rec, dx, dy);
%     [Ez_gxy, Ez_gxx] = gradient(Ez_gx, dx, dy);
%     [Ez_gyy, Ez_gyx] = gradient(Ez_gy, dx, dy);
% 
%     curl_curl_Ez = -1 .* Ez_gxx - Ez_gyy;
%     gama_k = curl_curl_Ez ./ Ez_tot_rec;
% 
%     eps_gama_k(:, :, ii) = real(gama_k) ./ (w ^ 2 * u_o);
%     sigma_gama_k(:, :, ii) = -1 * ((imag(gama_k)) ./ (w * u_o));
% end
% 
% eps_gama_k = sum(eps_gama_k, 3) ./ src_num;
% sigma_gama_k = sum(sigma_gama_k, 3) ./ src_num;
% 
% x_dash = x_dash(trunc_x_low : trunc_x_up);
% y_dash = y_dash(trunc_y_low : trunc_y_up);
% 
% [Axis_y, Axis_x] = ndgrid(x_dash, y_dash);
% 
% imagesc(Axis_x(:), Axis_y(:), eps_gama_k ./ eps_o); axis image; 
% figure; imagesc(Axis_x(:), Axis_y(:), sigma_gama_k); axis image; 