clear all
seed = 1;
randn('state',seed);
rand('state',seed);

Q = [0 -1; -1 0];
F = @(x) x(:)'*Q*x(:);
k_vec = [100000 100000];
weights = [1 1];

lower = [-1 -1];
upper = [3 3];
F_scaled = @(x,param_F) F(interpolate(x, lower, upper, k_vec));

[G, elem_weights, t_vec, mapping] = map_lattice_to_set(F_scaled, weights, k_vec,0.01);

param_min.param_F = [];
param_min.weights = elem_weights;

param_min.maxiter = 1000;
param_min.early_terminate_gap = 1e-7;

out_struct_min = minimize_submodular_set_regularized_pairwise_fw(G,param_min);
rho = out_struct_min.rho;

%% fix multiplier on constraint
alpha = 0.00802;
y = double(rho > alpha);
x_scaled = mapping(y);
x = interpolate(x_scaled, lower, upper, k_vec);

all_x1 = (0:100)/100 * (upper(1) - lower(1)) + lower(1);
all_x2 = (0:100)/100 * (upper(2) - lower(2)) + lower(2);

[X,Y] = meshgrid(all_x1, all_x2);
Z = zeros(size(X));
for ii=1:numel(Z)
    Z(ii) = F([X(ii) Y(ii)]) + alpha * dot(weights, [X(ii) Y(ii)]);
end

mesh(X,Y,Z);
xlabel('x');
ylabel('y');