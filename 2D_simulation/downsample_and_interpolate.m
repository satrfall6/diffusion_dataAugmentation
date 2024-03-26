function TD_Ez_mat_ds = downsample_and_interpolate(TD_Ez_mat, new_size)
    % Inputs:
    % TD_Ez_mat: The original [2999, 16, 16] matrix.
    % new_size: The new size for the first dimension, e.g., 256.
    
    % Trim the matrix if necessary (to [2000, 16, 16] in this case)
    if size(TD_Ez_mat, 1) > 2000
        TD_Ez_mat = TD_Ez_mat(1:2000, :, :);
    end
    
    % Prepare for downsampling
    original_size = size(TD_Ez_mat, 1); % Adjusted to be the trimmed size or original if less than 2000
    original_indices = linspace(1, original_size, original_size);
    new_indices = linspace(1, original_size, new_size);
    
    TD_Ez_mat_ds = zeros(new_size, 16, 16);
    
    % Since 'cubic' interpolation cannot be used with extrapolation,
    % and to ensure the function's robustness, we'll use 'linear' interpolation.
    method = 'linear';
    
    % Perform interpolation for each pixel in the 16x16 grid
    for i = 1:16
        for j = 1:16
            % Extract the current line to interpolate
            current_line = squeeze(TD_Ez_mat(:, i, j));
            
            % Ensure 'linear' interpolation to avoid extrapolation issues
            interpolated_line = interp1(original_indices, current_line, new_indices, method, 'extrap');
            
            % Assign the interpolated data back to the matrix
            TD_Ez_mat_ds(:, i, j) = interpolated_line;
        end
    end
end
