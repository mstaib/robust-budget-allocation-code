function [ param_F ] = submodular_fct_influence_adversary_build_param_from_problem( problem, eps )
%SUBMODULAR_FCT_INFLUENCE_ADVERSARY_BUILD_PARAM_FROM_PROBLEM Summary of this function goes here
%   Detailed explanation goes here

    %% setup the parameters needed for function evaluation
    param_F.S = problem.S;
    param_F.T = problem.T;

    % TODO: figure out when to use this somehow
    param_F.sparse = true;

    %param_F.x_centers = problem.x_median; % used only to fill in the missing values
    param_F.x_centers = problem.x_centers_empirical;
    
    param_F.k_mat = discretize_bipartite_problem(eps, problem);
    cannot_be_var = param_F.k_mat <= 1;
    problem.var_edges = problem.var_edges & ~cannot_be_var;
    
    %param_F.x_lower = problem.x_median; %problem.x_lower
    param_F.x_lower = problem.x_centers_empirical;
    
    param_F.x_upper = problem.x_upper;
    param_F.edges = problem.real_edges;
    param_F.var_edges = problem.var_edges;
    param_F.var_edges_inx = find(param_F.var_edges);
    
    param_F.k_vec = param_F.k_mat(problem.var_edges);
    param_F.x_lower_vec = param_F.x_lower(problem.var_edges);
    param_F.x_upper_vec = param_F.x_upper(problem.var_edges);
    
    param_F.var_memory_size = param_F.S * param_F.T;
    
    if param_F.sparse
        param_F.var_memory_size = nnz(param_F.var_edges);
        
        param_F.x_centers_sparse = sparse(param_F.x_centers);
        param_F.x_lower_sparse = sparse(param_F.x_lower);
        param_F.x_upper_sparse = sparse(param_F.x_upper);
    end
    
    
    if gpuDeviceCount > 0
        param_F.use_gpu = true;
    end
    param_F.use_gpu = false;
    
    if param_F.use_gpu && param_F.sparse
        % go ahead and create copies of some parameters on the GPU;
        param_F.x_centers_gpu_sparse = gpuArray(param_F.x_centers);
        param_F.var_edges_gpu = gpuArray(param_F.var_edges);
        param_F.var_edges_inx_gpu = gpuArray(find(param_F.var_edges));

        param_F.k_vec_gpu = gpuArray(param_F.k_vec);
        param_F.x_lower_vec_gpu_sparse = gpuArray(param_F.x_lower_vec);
        param_F.x_upper_vec_gpu_sparse = gpuArray(param_F.x_upper_vec);
        
        param_F.not_edges_inx_gpu = gpuArray(find(~param_F.edges));
        
        param_F.alpha_gpu = gpuArray(param_F.alpha);
        param_F.S_gpu = gpuArray(param_F.S);
        param_F.T_gpu = gpuArray(param_F.T);
    end
    
    if param_F.use_gpu% && ~param_F.sparse
        % go ahead and create copies of some parameters on the GPU;
        param_F.x_centers_gpu = gpuArray(single(param_F.x_centers));
        param_F.var_edges_gpu = gpuArray(single(param_F.var_edges));
        param_F.var_edges_inx_gpu = gpuArray(single(find(param_F.var_edges)));

        param_F.k_vec_gpu = gpuArray(single(param_F.k_vec));
        param_F.x_lower_vec_gpu = gpuArray(single(param_F.x_lower_vec));
        param_F.x_upper_vec_gpu = gpuArray(single(param_F.x_upper_vec));
        
        param_F.not_edges_inx_gpu = gpuArray(single(find(~param_F.edges)));
        
        param_F.alpha_gpu = gpuArray(single(param_F.alpha));
        param_F.S_gpu = gpuArray(single(param_F.S));
        param_F.T_gpu = gpuArray(single(param_F.T));
    end
end

