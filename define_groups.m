function [G_GL, G_S3] = define_groups(GL_generators, S3_generators)
    % DEFINE_GROUPS: Constructs structured GL(n)^3 and S3 subgroup representations
    % 
    % Inputs:
    %   n - Matrix size
    %   GL_generators - Cell array of 3x3 transformation matrices {U, V, W}
    %   S3_generators - Cell array of 1x3 permutations (e.g., {[2,3,1], [3,1,2]})
    %
    % Outputs:
    %   G_GL - Cell array of struct elements for GL(n)^3 subgroup
    %   G_S3 - Cell array of struct elements for S3 subgroup

    % Initialize groups
    num_GL_generators = length(GL_generators);
    num_S3_generators = length(S3_generators);
    
    G_GL = cell(1, num_GL_generators); % Preallocate G_GL
    G_S3 = cell(1, num_S3_generators); % Preallocate G_S3

    % Define GL(n)^3 Subgroup Elements
    for i = 1:num_GL_generators
        matrices = GL_generators{i}; % Get {U, V, W} for each generator
        G_GL{i} = struct('U', matrices{1}, 'V', matrices{2}, 'W', matrices{3});
    end

    % Define S3 Subgroup Elements
    for i = 1:num_S3_generators
        perm = S3_generators{i}; % Get the permutation for each generator
        G_S3{i} = struct('perm', perm);
    end
end
