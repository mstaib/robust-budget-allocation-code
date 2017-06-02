function [var_edges] = choose_var_edges(problem, eps, y_constraint, num_edges, y_mat)

if num_edges >= nnz(problem.real_edges)
    var_edges = problem.real_edges;
    return;
end

if nargin < 5
    % compute best budget with no adversary
    y_mat = best_budget_no_adversary(problem, eps, y_constraint);
end

problem.var_edges = false(size(problem.real_edges));
param_F = submodular_fct_influence_adversary_build_param_from_problem(problem, eps);

first_resulting_influence = NaN;

% F0 = influence_func_edges(param_F, problem.var_edges, y_mat);
% function val = F(edges_inx)
%     if isempty(edges_inx)
%         val = F0;
%     else
%         edges_logical = ismembc(1:numel(problem.real_edges), edges_inx);
%         val = influence_func_edges(param_F, edges_logical, y_mat);
%     end
% end

%F = @(edges) influence_func_edges(param_F, edges, y_mat);
%F0 = F(problem.var_edges);

eps = 0.1;
n = nnz(problem.real_edges);
s = round(n/num_edges * log(1/eps));

for kk=1:num_edges
    candidate_edges = problem.real_edges & ~problem.var_edges;
    sampled_candidate_edges_inx = randsample(find(candidate_edges), s);

    best_edge_inx = NaN;
    best_resulting_influence = Inf;
    
    %candidate_edge_inx = find(candidate_edges);
    for jj=1:length(sampled_candidate_edges_inx)
        edge_inx = sampled_candidate_edges_inx(jj);
        
        influence = marginal_gain(param_F, problem.var_edges, edge_inx, y_mat);
        if isnan(first_resulting_influence)
            first_resulting_influence = influence;
        end
        
        if influence < best_resulting_influence
            best_resulting_influence = influence;
            best_edge_inx = edge_inx;
        end
    end
    
    problem.var_edges(best_edge_inx) = true;
    param_F = update_param_var_edges(param_F, problem.var_edges);
end

var_edges = problem.var_edges;
fprintf('First influence: %0.4f, last influence: %0.4f\n', first_resulting_influence, best_resulting_influence);

end

function val = influence_func_edges(param_F, var_edges, y_mat)
    param_F = update_param_var_edges(param_F, var_edges);

    if param_F.use_gpu
        x_full_gpu = param_F.k_vec_gpu-1;
        val = submodular_fct_influence_adversary_vec_wrapper(x_full_gpu, y_mat, param_F);
    else
        x_full = param_F.k_vec-1;
        val = submodular_fct_influence_adversary_vec_wrapper(x_full, y_mat, param_F);
    end
end

function val = marginal_gain(param_F, var_edges, edge_inx, y_mat)
    var_edges(edge_inx) = true;
    param_F = update_param_var_edges(param_F, var_edges);
    
    if param_F.use_gpu %gpuDeviceCount > 0
        %x_zeros_gpu = gpuArray.zeros(size(param_F.k_vec_gpu));
        x_full_gpu = param_F.k_vec_gpu-1;
        val = submodular_fct_influence_adversary_vec_wrapper(x_full_gpu, y_mat, param_F);
    else
        x_full = param_F.k_vec-1;
        val = submodular_fct_influence_adversary_vec_wrapper(x_full, y_mat, param_F);
    end
end

function y_mat = best_budget_no_adversary(problem, eps, y_constraint)
    x_var_indices = randsample(find(problem.real_edges), 1);
    var_edges = false(size(problem.real_edges));
    var_edges(x_var_indices) = true;
    problem.var_edges = var_edges;
    
    param_F = submodular_fct_influence_adversary_build_param_from_problem(problem, eps);
    
    if param_F.use_gpu
        x_zeros_gpu = gpuArray.zeros(size(param_F.k_vec_gpu));
        [ y_mat, ~, ~, ~ ] = solve_influence_fixed_x( x_zeros_gpu, ones(param_F.S,1), param_F, y_constraint );
    else
        x_zeros = zeros(size(param_F.k_vec));
        [ y_mat, ~, ~, ~ ] = solve_influence_fixed_x( x_zeros, ones(param_F.S,1), param_F, y_constraint );
    end
end

function param_F = update_param_var_edges(param_F, var_edges)
    %% setup the parameters needed for function evaluation
    param_F.var_edges = var_edges;
    param_F.var_edges_inx = find(var_edges);

    param_F.k_vec = param_F.k_mat(param_F.var_edges_inx);
    param_F.x_lower_vec = param_F.x_lower(param_F.var_edges_inx);
    param_F.x_upper_vec = param_F.x_upper(param_F.var_edges_inx);
    
    if param_F.use_gpu %gpuDeviceCount > 0
        % go ahead and create copies of some parameters on the GPU;
        param_F.var_edges_gpu = gpuArray(single(var_edges));
        param_F.var_edges_inx_gpu = gpuArray(single(param_F.var_edges_inx));

        param_F.k_vec_gpu = gpuArray(single(param_F.k_vec));
        param_F.x_lower_vec_gpu = gpuArray(single(param_F.x_lower_vec));
        param_F.x_upper_vec_gpu = gpuArray(single(param_F.x_upper_vec));
    end
end