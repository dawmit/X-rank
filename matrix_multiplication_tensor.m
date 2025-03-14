function T = matrix_multiplication_tensor(n)
    % MATRIX_MULTIPLICATION_TENSOR Creates the matrix multiplication tensor using sptensor
    %
    % This function generates the matrix multiplication tensor for n×n matrices
    % defined as: Mn = Σ Ei,j ⊗ Ej,k ⊗ Ek,i over indices i,j,k = 1...n
    % where Ei,j is a matrix with a 1 at position (i,j) and zeros elsewhere.
    %
    % Uses the sptensor representation from the Tensor Toolbox.
    %
    % Input:
    %   n - The dimension of matrices being multiplied
    %
    % Output:
    %   T - The matrix multiplication tensor as an sptensor of size n^2 × n^2 × n^2
    
    % Import tensor toolbox (uncomment if needed)
    % import tensor.*
    
    % Get the total number of non-zero elements (n^3)
    nnz = n^3;
    
    % Preallocate arrays to store subscripts and values
    subs = zeros(nnz, 3);  % Each nonzero element has 3 subscripts (i,j,k)
    
    % Generate the subscripts for the matrix multiplication tensor
    idx = 1;
    for i = 1:n
        for j = 1:n
            for k = 1:n
                % Calculate the linear indices in the tensor
                % For Ei,j, the linear index in the flattened n×n matrix is (i-1)*n+j
                idx1 = (i-1)*n + j;  % Index for Ei,j
                idx2 = (j-1)*n + k;  % Index for Ej,k
                idx3 = (k-1)*n + i;  % Index for Ek,i
                
                % Store the subscripts
                subs(idx, :) = [idx1, idx2, idx3];
                idx = idx + 1;
            end
        end
    end
    
    % Create the sparse tensor using the subscripts and values
    T = sptensor(subs, 1);
    
    % The resulting tensor T represents the matrix multiplication operation
    % where T(a,b,c) = 1 if multiplying matrices with entries at positions
    % corresponding to a and b would result in a contribution to position c
end