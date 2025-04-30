# CP-ALS-QR-Cyclic

## Overview

This repository contains an implementation of the CP-ALS-QR-Cyclic algorithm for tensor decomposition with symmetry constraints. The algorithm is an enhanced version of the CANDECOMP/PARAFAC (CP) decomposition method using alternating least squares (ALS) with QR decomposition for improved numerical stability.

## Features

- **QR-based ALS Algorithm**: Uses QR decomposition instead of normal equations for improved numerical stability when solving least squares problems
- **Group-theoretic Symmetry Preservation**: Enforces symmetry constraints using group actions from GL(n)³ and S₃
- **Performance Timing**: Detailed timing information for each step of the algorithm
- **Tensor Support**: Works with various tensor types including dense tensors, sparse tensors, Kruskal tensors, and Tucker tensors
- **Configurable Parameters**: Multiple customization options including initialization methods, convergence criteria, and dimension ordering

## Dependencies

- MATLAB (R2018b or later recommended)
- MATLAB Tensor Toolbox (v3.1 or later)

## Installation

1. Download or clone this repository
2. Add the directory to your MATLAB path:
   ```matlab
   addpath('/path/to/CP-ALS-QR-Cyclic')
   ```
3. Ensure the MATLAB Tensor Toolbox is installed and added to your path
```

### Usage

### Basic Usage

```matlab
% Create a test tensor
X = tensor(randn([5, 5, 5]));

% Define group elements for symmetry
% Create GL(n)³ generators
GL_generators = {
    {eye(5), eye(5), eye(5)},  % Identity
    {2*eye(5), 0.5*eye(5), eye(5)}  % Simple scaling
};

% Create S₃ generators
S3_generators = {
    [2, 1, 3],  % Swap first two indices
    [1, 3, 2]   % Swap last two indices
};

% Generate group elements
[G_GL, G_S3] = define_groups(GL_generators, S3_generators);

% Compute CP decomposition with rank 3
[P, Uinit, output] = cp_als_qr_cyclic(X, 3, G_GL, G_S3);
```

### Advanced Usage

```matlab
% Specify custom parameters
opts = struct();
opts.tol = 1e-6;              % Convergence tolerance
opts.maxiters = 100;          % Maximum iterations
opts.dimorder = [3, 1, 2];    % Custom dimension processing order
opts.init = 'nvecs';          % SVD-based initialization
opts.printitn = 10;           % Print progress every 10 iterations
opts.errmethod = 'full';      % Error calculation method

% Create initial guess manually (optional)
n = 5;  % Tensor dimension
R = 3;  % Decomposition rank
len_GL = length(G_GL);
len_S3 = length(G_S3);
Uinit = cell(3,1);
Uinit{1} = rand(n^2, len_GL*len_S3*R);  % Initialize first factor
Uinit{2} = rand(n^2, len_GL*len_S3*R);  % Initialize second factor
Uinit{3} = rand(n^2, len_GL*len_S3*R);  % Initialize third factor

% Compute CP decomposition with custom options and initialization
[P, ~, output] = cp_als_qr_cyclic(X, R, G_GL, G_S3, 'init', Uinit, opts);
```

## Algorithm Details

### Inputs

- `X`: Input tensor (tensor, sptensor, ktensor, or ttensor)
- `R`: Rank of decomposition
- `G_GL`: Cell array of GL(n)³ group elements with fields U, V, W
- `G_S3`: Cell array of S₃ group elements with permutation field
- `varargin`: Optional parameters (see below)

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tol` | 1e-4 | Convergence tolerance on relative change in fit |
| `maxiters` | 50 | Maximum number of iterations |
| `dimorder` | 1:N | Order to loop through dimensions |
| `init` | 'random' | Initialization method ('random', 'nvecs', or cell array) |
| `printitn` | 1 | Print fit every n iterations; 0 for no printing |
| `errmethod` | 'fast' | Method for error calculation ('fast', 'full', 'lowmem') |

### Outputs

- `P`: Decomposed tensor as a ktensor
- `Uinit`: Initial guess for factor matrices
- `output`: Structure with additional information:
  - `params`: Input parameters
  - `iters`: Number of iterations performed
  - `relerr`: Relative error per iteration
  - `fit`: Final fit value (1 means perfect)
  - `times`: Timing information for different algorithm steps

## Symmetrization Process

The `symmetrize_tensor` function enforces symmetry constraints on the tensor factors through the following process:

1. Each rank-one component is symmetrized separately
2. For each component, symmetrized versions are accumulated by applying all group actions
3. The accumulated result is normalized by the group size
4. The normalized result is then distributed back across the orbit

This ensures the resulting tensor respects the symmetry constraints defined by the provided group actions.

### Group Theory Background

The symmetrization uses two types of group actions:

1. **GL(n)³ Actions**: Linear transformations that "sandwich" each factor matrix
   - For each factor A, B, C, transformations are applied as (U⋅A⋅V⁻¹, V⋅B⋅W⁻¹, W⋅C⋅U⁻¹)
   - These preserve the multilinear structure of the tensor

2. **S₃ Actions**: Permutations of the tensor indices
   - For example, permutation [2,3,1] transforms (A,B,C) to (C,A,B)
   - Odd permutations also require transposition of all matrices

The combination of these group actions enables enforcing various symmetry patterns in the decomposition.

## Performance Notes

- The algorithm uses QR decomposition instead of normal equations for improved numerical stability
- Timing information is collected for each major step:
  - `t_ttm`: Tensor-times-matrix calculations
  - `t_qrf`: QR factorization of factor matrices
  - `t_kr`: Khatri-Rao product calculations
  - `t_q0`: Q0 application
  - `t_back`: Backsolving with R matrices
  - `t_lamb`: Lambda calculations and normalization
  - `t_err`: Error calculation


## Contributors

- Irina Viviano
- Rachel Minster
