clc
clear

%load ./data/eps_mask_ellipse_5.mat
%load ./data/sigma_mask_ellipse_5.mat
load ./data/random_eps.mat
load ./data/random_sigma.mat

TD_Ez_mat = zeros(2999, 16, 16, 1000);
S_Ez_mat = zeros(16, 16, 1000);

%eps_mask = eps_mask_ellipse_5
%sigma_mask = sigma_mask_ellipse_5
eps_mask = random_eps
sigma_mask = random_sigma
case_num = size(eps_mask, 3);

for kk = 1 : case_num
    eps_gama = eps_mask(:, :, kk);
    sigma_gama = sigma_mask(:, :, kk);

    [TD_Ez_mat(:, :, :, kk), S_Ez_mat(:, :, kk)] = FDTD_solver(eps_gama, sigma_gama);

    fprintf('The %ith case is computed...\n', kk);
end

