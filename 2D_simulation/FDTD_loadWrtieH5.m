fprintf('*******************************************************\n');
partition = getenv('PARTITION');
%partition = '1';
partition_emp = partition;
fixed = getenv('FIXED');
%fixed = 'fixed';
if strcmp(fixed, 'fixed')
    partition_emp = '1';
end 
partition_int = int8(str2double(partition));
%disp(['Now running partition: ', partition]);
fprintf('Now running partition: %s %s \n', fixed, partition);
%%
% Incorporating 'partition' into the filenames
stroke_per_file = sprintf('./data/stroke_per_%s_%s.h5', fixed, partition); % Specify your HDF5 file name
stroke_per_info = h5info(stroke_per_file);
empty_per_file = sprintf('./data/empty_per_%s_%s.h5', fixed, partition_emp); % Specify your HDF5 file name
empty_per_info = h5info(empty_per_file);
stroke_con_file = sprintf('./data/stroke_con_%s_%s.h5', fixed, partition); % Specify your HDF5 file name
stroke_con_info = h5info(stroke_con_file);
empty_con_file = sprintf('./data/empty_con_%s_%s.h5', fixed, partition_emp); % Specify your HDF5 file name
empty_con_info = h5info(empty_con_file);
%%
FDTD_filename = sprintf('./data/FDTD_stroke_%s_%s.h5', fixed, partition); % Specify your HDF5 file name
if strcmp(fixed, 'fixed')
    FDTD_empty_filename = sprintf('./data/FDTD_empty_%s.h5', fixed); % Specify your HDF5 file name
else
    FDTD_empty_filename = sprintf('./data/FDTD_empty_%s_%s.h5', fixed, partition); % Specify your HDF5 file name
end

% Get info for FDTD file
if isfile(FDTD_filename)
    FDTD_info = h5info(FDTD_filename);
else
    FDTD_info = struct('Datasets', []);  % No datasets if file doesn't exist
end
cases_in_partition = length(stroke_per_info.Datasets); % divide into 10 partitions
%start_idx = (partition_int-1)*cases_in_partition + 1;
%disp(['starting with index: ', start_idx]);
%fprintf('starting with index: %s \n', string(start_idx));
fprintf('*******************************************************\n');

% List all datasets at the current level
for i = 1:cases_in_partition
    key_stroke = strcat('/', stroke_per_info.Datasets(i).Name);
    key_empty = strcat('/', empty_per_info.Datasets(i).Name);

    fprintf('Processing_Exp: %s\n', key_stroke);
    % Optionally, read data from dataset
    % !!only when read h5 in matlab need to transpose, read in python is fine
    mask_per = h5read(stroke_per_file, key_stroke);
    mask_per = mask_per';
    empty_per = h5read(empty_per_file, key_empty);
    empty_per = empty_per';
    mask_con = h5read(stroke_con_file, key_stroke);
    mask_con = mask_con';
    empty_con = h5read(empty_con_file, key_empty);
    empty_con = empty_con';

    disp(size(mask_per));

    % Check if dataset already exists in FDTD file
    datasetExists = any(arrayfun(@(x) strcmp(x.Name, stroke_per_info.Datasets(i).Name), FDTD_info.Datasets));
    
    % If the dataset does not exist, create it, write data, and add attribute
    if ~datasetExists
        fprintf('Dataset does not exist in FDTD file. Creating dataset...\n');
        
        TD_Ez_mat = zeros(2999, 16, 16);
        S_Ez_mat = zeros(16, 16);
        [TD_Ez_mat(:, :, :), S_Ez_mat(:, :)] = FDTD_solver(mask_per, mask_con);
        TD_Ez_mat_ds = downsample_and_interpolate(TD_Ez_mat, 256);
        h5create(FDTD_filename, key_stroke, size(TD_Ez_mat_ds));  % Use dynamic sizing based on actual data
        fprintf('Downsampled array size:\n');

        h5write(FDTD_filename, key_stroke, TD_Ez_mat_ds);
        h5writeatt(FDTD_filename, key_stroke, 'description', 'RandomHead_td');
        disp(size(TD_Ez_mat_ds));


        if strcmp(fixed, 'fixed')
            if i==1 & partition=='1'
                TD_Ez_mat = zeros(2999, 16, 16);
                S_Ez_mat = zeros(16, 16);
                [TD_Ez_mat(:, :, :), S_Ez_mat(:, :)] = FDTD_solver(empty_per, empty_con);
                TD_Ez_mat_ds = downsample_and_interpolate(TD_Ez_mat, 256);
                disp(size(TD_Ez_mat_ds));

                h5create(FDTD_empty_filename, key_empty, size(TD_Ez_mat_ds));  % Use dynamic sizing based on actual data
                h5write(FDTD_empty_filename, key_empty, TD_Ez_mat_ds);
                h5writeatt(FDTD_empty_filename, key_empty, 'description', 'RandomHead_td');
            else
                fprintf('Running on fixed head passed all other empty FDTDs\n')
                fprintf('=======================================================\n');

            end
        else
            TD_Ez_mat = zeros(2999, 16, 16);
            S_Ez_mat = zeros(16, 16);
            [TD_Ez_mat(:, :, :), S_Ez_mat(:, :)] = FDTD_solver(empty_per, empty_con);
            TD_Ez_mat_ds = downsample_and_interpolate(TD_Ez_mat, 256);
            disp(size(TD_Ez_mat_ds));
            h5create(FDTD_empty_filename, key_empty, size(TD_Ez_mat_ds));  % Use dynamic sizing based on actual data
            h5write(FDTD_empty_filename, key_empty, TD_Ez_mat_ds);
            h5writeatt(FDTD_empty_filename, key_empty, 'description', 'RandomHead_td');
            fprintf('=======================================================\n');
        end
    else
        fprintf('Dataset already exists in FDTD file. Skipping creation.\n');
        fprintf('=======================================================\n');
    end
end
fprintf('Partition %s done!\n', partition);
%%

%imagesc(Axis_x(:), Axis_y(:), eps_gama);axis image
%hold on
%scatter(Probes_Tx(:, 2), Probes_Tx(:, 1), 'r*');axis image

