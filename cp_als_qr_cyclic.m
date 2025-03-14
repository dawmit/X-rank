function [P,Uinit,output] = cp_als_qr_cyclic(X,R, G_GL, G_S3, varargin)
%CP_ALS_QR Compute a CP decomposition of any type of tensor.
%
%   M = CP_ALS(X,R) computes an estimate of the best rank-R
%   CP model of a tensor X using an alternating least-squares
%   algorithm.  The input X can be a tensor, sptensor, ktensor, or
%   ttensor. The result P is a ktensor.
%
%   M = CP_ALS(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init' - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%
%   [M,U0] = CP_ALS(...) also returns the initial guess.
%
%   [M,U0,out] = CP_ALS(...) also returns additional output that contains
%   the input parameters.
%
%   Note: The "fit" is defined as 1 - norm(X-full(P))/norm(X) and is
%   loosely the proportion of the data described by the CP model, i.e., a
%   fit of 1 is perfect.
%
%   NOTE: Updated in various minor ways per work of Phan Anh Huy. See Anh
%   Huy Phan, Petr Tichavsk?, Andrzej Cichocki, On Fast Computation of
%   Gradients for CANDECOMP/PARAFAC Algorithms, arXiv:1204.1586, 2012.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   M = cp_als(X,2);
%   M = cp_als(X,2,'dimorder',[3 2 1]);
%   M = cp_als(X,2,'dimorder',[3 2 1],'init','nvecs');
%   U0 = {rand(5,2),rand(4,2),[]}; %<-- Initial guess for factors of P
%   [M,U0,out] = cp_als(X,2,'dimorder',[3 2 1],'init',U0);
%   M = cp_als(X,2,out.params); %<-- Same params as previous run
%
%   See also KTENSOR, TENSOR, SPTENSOR, TTENSOR.
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.
%
% Adapted to use a QR to solve the LS problems instead of normal
% equations, also includes timing of iteration parts, and outputs relative
% error and fit. Used in 'CP Decomposition for Tensors via Alternating Least Squares with QR
% Decomposition'
% - Adapted by Irina Viviano & Rachel Minster, 2021




%% Extract number of dimensions and norm of X.
N = ndims(X);
fprintf("Dimension %d\n",N);
normX = norm(X);
fprintf("Norm X %d\n", normX);
len_GL = length(G_GL);
len_S3 = length(G_S3);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParameter('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParameter('printitn',1,@isscalar);
params.addParameter('errmethod','fast',@(x) ismember(x,{'fast','full','lowmem'}));
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;
errmethod = params.Results.errmethod;


%% Error checking 

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(2:end)
        if ~isequal(size(Uinit{n}),[size(X,n) (len_S3*len_GL*R)])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    % Observe that we don't need to calculate an initial guess for the
    % first index in dimorder because that will be solved for in the first
    % inner iteration.
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = rand(size(X,n),(len_S3*len_GL*R));
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = nvecs(X,n,(len_S3*len_GL*R));
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit = 0;
disp('Init Uinit (during set up)');
disp(U);
if printitn>0
  fprintf('\nCP_ALS_QR:\n');
end


%% Main Loop: Iterate until convergence

%%% Changes for cp_als_qr start here: %%%

%%% Initialize a cell array Qs and Rs to hold decompositions of factor matrices. %%%
Qs = cell(N,1); %%% The Kronecker product of these tells us part of the Q of the Khatri-Rao product. %%%
Rs = cell(N,1); %%% The Khatri-Rao product of these tells us the rest of Q and the R of the Khatri-Rao product. %%%
%%% Compute economy-sized QR decomposition. %%%
for i = 1:N
    if ~isempty(U{i})
        [Qs{i}, Rs{i}] = qr(U{i},0); 
    end
end
   

for iter = 1:maxiters
    fprintf('Iteration: %d\n', iter);
    t_ttm = 0; % TTM
    t_qrf = 0; % QR of factor matrices
    t_kr = 0; % Computing Q0
    t_q0 = 0; % Applying Q0
    t_back = 0;
    t_lamb = 0;
    t_err = 0;
        
    fitold = fit;
        
    % Iterate over all N modes of the tensor
    for n = dimorder(1:end)
        %%% Compute the QR of the Khatri-Rao product of Rs. %%%
        %%% First compute the Khatri-Rao product on all modes but n. %%%
        tic;  M = khatrirao(Rs{[1:n-1,n+1:N]},'r'); t = toc; t_kr = t_kr + t;


        %%% Compute the explicit QR factorization.
        tic;  [Q0,R0] = qr(M,0); t = toc; t_kr = t_kr + t;
        %%% TTM on all modes but mode n. %%%
        tic; Y = ttm(X,Qs,-n,'t'); t = toc; t_ttm = t_ttm + t;


        %%% Now multiply by Q0 on the right. %%%
        
        if isa(Y,'ktensor')
            %%% For a ktensor: %%%
            %%% Save all the factor matrices of Y in a cell array. %%%
            %%% Then, we can compute the Khatri Rao product in one line.%%%
            K = cell(N,1);
            disp('This is when Y is a k-tensor');
            for k = 1:N
                %if k ~= n
                    K{k} = Y.U{k};
                %end
            end
            
            %%% Apply Q0
            tic; Z = Y.U{n} * (khatrirao(K{[1:n-1,n+1:N]},'r')' * Q0); t = toc; t_q0 = t_q0 + t;

            %%% Calculate updated factor matrix by backsolving with R0' and Z. %%%
            tic; U{n} = double(Z) / R0'; t = toc; t_back = t_back + t;
            
        else
            %%% For any other tensor: %%%
            %%% Apply Q0 %%%
            tic; Z = tenmat(Y,n) * Q0; t = toc; t_q0 = t_q0 + t;

            %%% Calculate updated factor matrix by backsolving with R0' and Z. %%%
            tic; U{n} = double(Z) / R0'; t = toc; t_back = t_back + t;


        end
               
        % Normalize each vector to prevent singularities in coefmatrix
        tic;
        if iter == 1
            lambda = sqrt(sum(U{n}.^2,1))'; %2-norm
        else
            lambda = max( max(abs(U{n}),[],1), 1 )'; %max-norm
        end 
        
        Unew = bsxfun(@rdivide, U{n}, lambda'); t = toc; t_lamb = t_lamb + t;
        U{n} = Unew;
        [U{1}, U{2}, U{3}] = symmetrize_tensor(U{1}, U{2}, U{3}, N, R, G_GL, G_S3);

        P = ktensor(lambda, U);
        %%% Recompute QR factorization for updated factor matrix. %%%
        tic; [Qs{n}, Rs{n}] = qr(U{n},0); t_qrf = toc;

    end
    %%% Changes for cp_als_qr end here. %%%

    tic;
    if normX == 0
        disp('Was Rescaled due to normX == 0');
       Rscaled = lambda.*R0';
       prod = U{dimorder(end)}*Rscaled;
       iprod = Z(:)'*prod(:);
       fit = norm(P)^2 - 2 * iprod;
    else
       switch errmethod
            case 'fast'
                % fast inner product calculation
                Rscaled = R0.*lambda';
                prod = U{dimorder(end)}*Rscaled';
                iprod = Z(:)'*prod(:); 

                % fast norm(P) calculation: < Lambda R0^T R0 Lambda, Rs{N}^T Rs{N} >
                RscaledGram = Rscaled'*Rscaled;
                RnGram = Rs{dimorder(end)}'*Rs{dimorder(end)};
                normPsq = RscaledGram(:)'*RnGram(:);
                
                normresidual = sqrt( abs(normX^2 + normPsq - 2 * iprod) );
            case 'full'
                normresidual = norm(full(X) - full(P));
            case 'lowmem'
                normresidual = normdiff(X,P); 
        end
        fit = 1 - (normresidual / normX); %fraction explained by model
        %%% Change this to just be relative error to see the error go down. %%%
        rel_err(iter,:) = normresidual / normX; 
    end
    fitchange = abs(fitold - fit); t = toc; t_err = t_err + t;
        
    % Check for convergence
    if (iter > 1) && (fitchange < fitchangetol)
        flag = 0;
    else
        flag = 1;
    end
        
    %%% If the fit is NaN, just stop the process. %%%
    if isnan(fit)
        break;
    end
    
    if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
        fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
    end

    % Check for convergence
    if (flag == 0)
        break;
    end     
    
    times(iter,:) = [t_ttm, t_qrf, t_kr, t_q0, t_back, t_lamb, t_err];
end   


%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
disp('Arranged final tensor');
disp(P);
% Fix the signs
P = fixsigns(P);
disp('Fixed signs');
disp(P);


if printitn>0
    if normX == 0
        fit = norm(P)^2 - 2 * innerprod(X,P);
    else
        switch errmethod
            case 'fast'
                Rscaled = lambda.*R0';
                prod = U{dimorder(end)}*Rscaled;
                iprod = Z(:)'*prod(:);

                % fast norm(P) calculation: < Lambda R0^T R0 Lambda, Rs{N}^T Rs{N} >
                RscaledGram = Rscaled*Rscaled';
                RnGram = Rs{dimorder(end)}'*Rs{dimorder(end)};
                normPsq = RscaledGram(:)'*RnGram(:);
                normresidual = sqrt( abs(normX^2 + normPsq - 2 * iprod) );
            case 'full'
                normresidual = norm(full(X) - full(P));
            case 'lowmem'
                normresidual = normdiff(X,P);
        end
        fit = 1 - (normresidual / normX); %fraction explained by model
        rel_err(iter,:) = normresidual / normX;
    end
  fprintf(' Final f = %e \n', fit);
end



output = struct;
output.params = params.Results;
output.iters = iter;
output.relerr = rel_err; %%% Add a rel_err vector to output
output.fit = fit;
output.times = times;

end


function [A_sym, B_sym, C_sym] = symmetrize_tensor(A, B, C, n, R, G_GL, G_S3)
    % SYMMETRIZE_TENSOR
    % Symmetrizes tensor factors using group actions from GL(n)^3 and S3
    % Now with proper handling of odd permutations
    %
    % Inputs:
    %   A, B, C - Tensor factors (n^2 x (R*|G_GL|*|G_S3|))
    %   n - Matrix dimension (matrices are n x n)
    %   R - Number of rank-one tensor blocks
    %   G_GL - Cell array of GL(n)^3 group elements with fields U, V, W
    %   G_S3 - Cell array of S3 group elements with field perm
    %
    % Outputs:
    %   A_sym, B_sym, C_sym - Symmetrized tensor factors
    
    num_GL = length(G_GL);
    num_S3 = length(G_S3);
    
    A_sym = A;
    B_sym = B;
    C_sym = C;
    
    % Process each rank-one tensor block
    for i = 1:R
        % Get indices for the current block
        block_start = (i-1) * num_GL * num_S3 + 1;
        block_end = i * num_GL * num_S3;
        block_indices = block_start:block_end;
        
        fprintf('Processing block %d (indices %d:%d)\n', i, block_start, block_end);
        
        % Step 1: Accumulate symmetrized versions
        A_accum = zeros(n, n);
        B_accum = zeros(n, n);
        C_accum = zeros(n, n);
        
        for j = 1:num_GL
            for k = 1:num_S3
                % Calculate index within the current block
                idx = (j-1) * num_S3 + k;
                global_idx = block_indices(idx);
                
                % Get group elements
                g_gl = G_GL{j};  % GL(n)^3 element
                g_s3 = G_S3{k};  % S3 permutation element
                
                % Reshape vector factors to matrices
                disp(n);
                matrixA = reshape(A(:, global_idx), [n, n]);
                matrixB = reshape(B(:, global_idx), [n, n]);
                matrixC = reshape(C(:, global_idx), [n, n]);
                
                % Check if the permutation is odd
                is_odd_perm = is_odd_permutation(g_s3.perm);
                
                % First apply S3 permutation - reorder the factors
                permuted_indices = g_s3.perm;
                permuted_matrices = {matrixA, matrixB, matrixC};
                
                % Get the matrices in permuted order
                A_perm = permuted_matrices{permuted_indices(1)};
                B_perm = permuted_matrices{permuted_indices(2)};
                C_perm = permuted_matrices{permuted_indices(3)};
                
                % If permutation is odd, transpose all factors
                if is_odd_perm
                    A_perm = A_perm';
                    B_perm = B_perm';
                    C_perm = C_perm';
                end
                
                % Then apply GL(n)^3 transformation (sandwiching action)
                U_adj = g_gl.U;
                V_adj = g_gl.V;
                W_adj = g_gl.W;
                
                A_transformed = U_adj * A_perm / (V_adj);
                B_transformed = V_adj * B_perm / (W_adj);
                C_transformed = W_adj * C_perm / (U_adj);
                
                % Accumulate transformed components
                A_accum = A_accum + A_transformed;
                B_accum = B_accum + B_transformed;
                C_accum = C_accum + C_transformed;
            end
        end
        
        % Normalize by the group size
        A_accum = A_accum / (num_GL * num_S3);
        B_accum = B_accum / (num_GL * num_S3);
        C_accum = C_accum / (num_GL * num_S3);
        
        % Step 2: Distribute symmetrized tensors back to the entire orbit
        for j = 1:num_GL
            for k = 1:num_S3
                % Calculate index within the current block
                idx = (j-1) * num_S3 + k;
                global_idx = block_indices(idx);
                
                % Get group elements
                g_gl = G_GL{j};  % GL(n)^3 element
                g_s3 = G_S3{k};  % S3 permutation element
                
                % Create copies of the accumulated matrices for this orbit element
                A_orbit = A_accum;
                B_orbit = B_accum;
                C_orbit = C_accum;
                
                % First apply S3 permutation (inverse)
                permuted_indices = g_s3.perm;
                
                % Determine inverse permutation
                inv_perm = zeros(1,3);
                for p = 1:3
                    inv_perm(permuted_indices(p)) = p;
                end
                
                % Check if the inverse permutation is odd
                is_odd_perm = is_odd_permutation(inv_perm);
                
                % If inverse permutation is odd, transpose accumulated matrices
                if is_odd_perm
                    A_orbit = A_orbit';
                    B_orbit = B_orbit';
                    C_orbit = C_orbit';
                end
                
                % Prepare components in permuted order
                orbit_components = {A_orbit, B_orbit, C_orbit};
                A_perm = orbit_components{inv_perm(1)};
                B_perm = orbit_components{inv_perm(2)};
                C_perm = orbit_components{inv_perm(3)};
                
                % Then apply GL(n)^3 transformation
                U_adj = g_gl.U;
                V_adj = g_gl.V;
                W_adj = g_gl.W;
                
                % Apply inverse transformations
                A_final = U_adj * A_perm / (V_adj);
                B_final = V_adj * B_perm / (W_adj);
                C_final = W_adj * C_perm / (U_adj);
                
                % Store results
                A_sym(:, global_idx) = reshape(A_final, [], 1);
                B_sym(:, global_idx) = reshape(B_final, [], 1);
                C_sym(:, global_idx) = reshape(C_final, [], 1);
            end
        end
    end
end

function is_odd = is_odd_permutation(perm)
    % Determine if a permutation is odd by counting inversions
    % A permutation is odd if it has an odd number of inversions
    n = length(perm);
    inversion_count = 0;
    
    % Only loop up to n-1 since the last element has no elements after it
    for i = 1:n-1
        for j = i+1:n
            if perm(i) > perm(j)
                inversion_count = inversion_count + 1;
            end
        end
    end
    
    is_odd = mod(inversion_count, 2) == 1;
end

function [A, B, C] = enforce_C3_symmetry(A, B, C, R)
%ENFORCE_C3_SYMMETRY Enforces cyclic C3 symmetry on factor matrices.
%   Inputs: A, B, C are factor matrices of size N x R.
%           R is the rank of the decomposition (must be a multiple of 3).
%   Outputs: Symmetrized factor matrices A, B, C.


    % Enforce symmetry over blocks of 3 rank-one tensors
    for r = 1:3:R
        % Extract the current block of 3 rank-one tensors
        A1 = A(:, r);   B1 = B(:, r);   C1 = C(:, r);
        A2 = A(:, r+1); B2 = B(:, r+1); C2 = C(:, r+1);
        A3 = A(:, r+2); B3 = B(:, r+2); C3 = C(:, r+2);

        % Symmetrize the block by averaging over group actions
        A(:, r)   = (A1 + C2 + B3) / 3; %A1
        B(:, r)   = (B1 + A2 + C3) / 3; %B1
        C(:, r)   = (C1 + B2 + A3) / 3; %C1

        A(:, r+1) = (A2 + C3 + B1) / 3; %A2
        B(:, r+1) = (B2 + A3 + C1) / 3; %B2
        C(:, r+1) = (C2 + B3 + A1) / 3; %C2

        A(:, r+2) = (A3 + C1 + B2) / 3; %A3
        B(:, r+2) = (B3 + A1 + C2) / 3; %B3
        C(:, r+2) = (C3 + B1 + A2) / 3; %C3
    end
end

function [A_sym, B_sym, C_sym] = enforce_Dn_symmetry(A, B, C, n, R)
    % ENFORCE_DIHEDRAL_SYMMETRY Enforces dihedral group symmetry on rank-one tensor components.
    %
    % Inputs:
    %   A, B, C: Vectors of size n^2 representing the components of the rank-one tensor.
    %   n: Size of the matrix (n x n) for reshaping.
    %
    % Outputs:
    %   A_sym, B_sym, C_sym: Symmetrized components after enforcing dihedral symmetry.


    % Create the antidiagonal (flip) matrix for reflection
    flip_matrix = fliplr(eye(n));

    % Create the diagonal matrix with nth roots of unity for rotation
    %roots_unity = exp(2i * pi * (0:n-1) / n); % nth roots of unity
    %roots_diag = roots_unity .^ (1:n);
    %rotation_matrix = diag(roots_diag);
    rotation_matrix = [-1,0;0,1];
    %disp(rotation_matrix);

    % Initialize symmetrized matrices
    A_sym = A; B_sym = B; C_sym = C;

    for r = 1:4:(4*R)  % Process each block of 4 rank-one tensors
        % Extract the current block
        A1 = reshape(A(:, r), [n,n]);   B1 = reshape(B(:, r), [n,n]);   C1 = reshape(C(:, r), [n,n]);
        A2 = reshape(A(:, r+1), [n,n]); B2 = reshape(B(:, r+1), [n,n]); C2 = reshape(C(:, r+1), [n,n]);
        A3 = reshape(A(:, r+2), [n,n]); B3 = reshape(B(:, r+2), [n,n]); C3 = reshape(C(:, r+2), [n,n]);
        A4 = reshape(A(:, r+3), [n,n]); B4 = reshape(B(:, r+3), [n,n]); C4 = reshape(C(:, r+3), [n,n]);


        % Compute symmetrized blocks
        A_sym(:, r)   = reshape((A1 + flip_matrix*A2*flip_matrix + rotation_matrix*A3*rotation_matrix' + flip_matrix*rotation_matrix*A4*rotation_matrix'*flip_matrix) / 4,[],1);
        B_sym(:, r)   = reshape((B1 + flip_matrix*B2*flip_matrix + rotation_matrix*B3*rotation_matrix' + flip_matrix*rotation_matrix*B4*rotation_matrix'*flip_matrix) / 4,[],1);
        C_sym(:, r)   = reshape((C1 + flip_matrix*C2*flip_matrix + rotation_matrix*C3*rotation_matrix' + flip_matrix*rotation_matrix*C4*rotation_matrix'*flip_matrix) / 4,[],1);

        A_sym(:, r+1)   = reshape((A2 + flip_matrix*A3*flip_matrix + rotation_matrix*A4*rotation_matrix' + flip_matrix*rotation_matrix*A1*rotation_matrix'*flip_matrix) / 4,[],1);
        B_sym(:, r+1)   = reshape((B2 + flip_matrix*B3*flip_matrix + rotation_matrix*B4*rotation_matrix' + flip_matrix*rotation_matrix*B1*rotation_matrix'*flip_matrix) / 4,[],1);
        C_sym(:, r+1)   = reshape((C2 + flip_matrix*C3*flip_matrix + rotation_matrix*C4*rotation_matrix' + flip_matrix*rotation_matrix*C1*rotation_matrix'*flip_matrix) / 4,[],1);

        A_sym(:, r+2)   = reshape((A3 + flip_matrix*A4*flip_matrix + rotation_matrix*A1*rotation_matrix' + flip_matrix*rotation_matrix*A2*rotation_matrix'*flip_matrix) / 4,[],1);
        B_sym(:, r+2)   = reshape((B3 + flip_matrix*B4*flip_matrix + rotation_matrix*B1*rotation_matrix' + flip_matrix*rotation_matrix*B2*rotation_matrix'*flip_matrix) / 4,[],1);
        C_sym(:, r+2)   = reshape((C3 + flip_matrix*C4*flip_matrix + rotation_matrix*C1*rotation_matrix' + flip_matrix*rotation_matrix*C2*rotation_matrix'*flip_matrix) / 4,[],1);

        A_sym(:, r+3)   = reshape((A4 + flip_matrix*A1*flip_matrix + rotation_matrix*A2*rotation_matrix' + flip_matrix*rotation_matrix*A3*rotation_matrix'*flip_matrix) / 4,[],1);
        B_sym(:, r+3)   = reshape((B4 + flip_matrix*B1*flip_matrix + rotation_matrix*B2*rotation_matrix' + flip_matrix*rotation_matrix*B3*rotation_matrix'*flip_matrix) / 4,[],1);
        C_sym(:, r+3)   = reshape((C4 + flip_matrix*C1*flip_matrix + rotation_matrix*C2*rotation_matrix' + flip_matrix*rotation_matrix*C3*rotation_matrix'*flip_matrix) / 4,[],1);
    end
end