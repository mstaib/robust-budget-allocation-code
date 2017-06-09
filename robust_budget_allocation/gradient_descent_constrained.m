function [xmin, out_struct] = gradient_descent_constrained(R_cell,param)
%MINIMIZE_SUBMODULAR_CONSTRAINED Summary of this function goes here
%   Detailed explanation goes here

% F takes two arguments (x,param_F)
% we assume param_F is passed to us as param.param_F

num_iters = 100;
learning_rate = 0.01;
tol = 1e-5;

n = length(R_cell);
x = ones(n,1);
x_hist = x;
fval = submodular_fct_influence_adversary_vec_wrapper(x, param.y_mat, param.param_F);


y_length = length(param.y_mat);
y_mat_diag = diag(param.y_mat);

% yalmip
y = sdpvar(n,1); x_proj = sdpvar(n,1); total = sdpvar(1); constraints = sdpvar(n,1);
Constraints = [x_proj >= param.param_F.x_lower_vec, x_proj <= param.param_F.x_upper_vec];
for ii=1:n
    Constraints = [Constraints, constraints(ii) >= R_cell{ii}(x_proj(ii))];
end
Constraints = [Constraints, sum(constraints) <= param.constraint];

Objective = norm(x_proj - y, 2);

options = sdpsettings('solver','mosek');
P = optimizer(Constraints, Objective, options, y, x_proj);
%P{x_after_step}


for iter=1:num_iters
    this_iter = tic;
    
    x_mat = param.param_F.x_lower_sparse;
    x_mat(param.param_F.var_edges) = interpolate(x, param.param_F.x_lower_vec, param.param_F.x_upper_vec, param.param_F.k_vec);
    grad_x_mat = submodular_fct_influence_adversary_x_grad_sparse(spfun(@log, x_mat), y_mat_diag);
    grad_x = grad_x_mat(param.param_F.var_edges);
    
    x_after_step = (x - learning_rate * grad_x);
    
    x_proj = P{x_after_step};
%     cvx_solver mosek
%     cvx_begin quiet
%         variable x_proj(n)
%         minimize norm(x_proj - x_after_step, 2)
%         total = 0;
%         for ii=1:n
%             total = total + R_cell{ii}(x_proj(ii));
%         end
%         total <= param.constraint;
%         x_proj >= param.param_F.x_lower_vec;
%         x_proj <= param.param_F.x_upper_vec;
%     cvx_end
    
    x = double(x_proj);
    
    x_hist = [x_hist x];
    fval = [fval submodular_fct_influence_adversary_vec_wrapper(x, param.y_mat, param.param_F)];
    
    iter_time(iter) = toc(this_iter);
    
    if iter > 1 && norm(x - x_hist(:,end-1), 2) <= tol
        break
    end
end

out_struct.iter_time = iter_time;

xmin = x;

end