function [xmin,Fmin,out_struct,certified_suboptimality_gap] = minimize_submodular_constrained(F,R_cell,oracle_R,param)
%MINIMIZE_SUBMODULAR_CONSTRAINED Summary of this function goes here
%   Detailed explanation goes here

% F takes two arguments (x,param_F)
% we assume param_F is passed to us as param.param_F

if ~isfield(param, 'weights')   
    fprintf('Computing weights...');
    weights_tic = tic;
    param.weights = compute_weights_from_regularizer(R_cell, param);
    weights_time = toc(weights_tic);
    fprintf('took %0.1f seconds\n', weights_time);
end

out_struct = minimize_submodular_regularized_pairwise_fw(F,param);

multiplier_timer = tic;
[xmin,Fmin,certified_suboptimality_gap] = find_optimal_continuous_x(F, oracle_R, param, out_struct.rho);
out_struct.multiplier_time = toc(multiplier_timer);

% so we don't need to recompute this every time
out_struct.weights = param.weights;

end

function [xmin,Fmin,certified_suboptimality_gap] = find_optimal_continuous_x(F, oracle_R, param, rho)
    function val = oracle_H(x)
        val = F(x,param.param_F);
    end
    function val = oracle_R_unscaled(x)
        x_ctns = interpolate(x, param.param_F.x_lower_vec, param.param_F.x_upper_vec, param.param_F.k_vec);
        val = oracle_R(x_ctns);
    end

    [xmin_low, xmin_high] = greedy_sort_search(rho, @oracle_R_unscaled, param.constraint);
    if xmin_low == xmin_high
        xmin = xmin_low;
        Fmin = oracle_H(xmin);
        certified_suboptimality_gap = 0;
        return;
    end    
    
    % we didn't prove anything about this, but try to continuously increase
    % x until we're right at the constraint boundary
    tol = 1e-3;
    t_low = 0;
    t_high = 1;
    while t_high - t_low > tol
        t_mid = (t_low + t_high)/2;
        x = (1 - t_mid)*xmin_low + t_mid*xmin_high;
        if oracle_R_unscaled(x) <= param.constraint
            t_low = t_mid;
        else
            t_high = t_mid;
        end
    end
    
    % finally return xmin and Fmin
    xmin = (1 - t_low)*xmin_low + t_low*xmin_high;
    Fmin = oracle_H(xmin);
    certified_suboptimality_gap.xmin_high_bound = Fmin - oracle_H(xmin_high);
    
    % we can also compute the gap lambda^* times (B - R(x))
    ind = find(floor(xmin_low) > 0);
    thresholds = arrayfun(@(ii) rho{ii}(floor(xmin_low(ii))), ind);
    lambda = min(thresholds);
    certified_suboptimality_gap.lagrangian_bound = oracle_H(xmin) - oracle_H(xmin_low)...
        + lambda * (param.constraint - oracle_R_unscaled(xmin_low));
    certified_suboptimality_gap.lagrangian_bound_discrete = lambda * (param.constraint - oracle_R_unscaled(xmin_low));
    certified_suboptimality_gap.lambda = lambda;
end

function [xmin_low, xmin_high] = greedy_sort_search(rho, oracle_R_unscaled, constraint)
    n = length(rho);
    num_total_rhos = sum(cellfun(@(x) length(x), rho));
    
    memory_cap = 1e7;
    
    most_vecs_at_a_time = floor(memory_cap / n);
    most_vecs_at_a_time = max(most_vecs_at_a_time, 3); %need at least three to do binary search
    most_vecs_at_a_time = min(most_vecs_at_a_time, num_total_rhos); %we need no more than n
    
    skip = floor( (num_total_rhos-1) / (most_vecs_at_a_time-1) );
    
    start_inx = 1;
    end_inx = num_total_rhos;
    
    while end_inx - start_inx > 1
        x_all = greedy_sort_subsample(rho, start_inx, end_inx, skip);
        
        [lower_inx, upper_inx] = binary_feasibility_search(x_all, oracle_R_unscaled, constraint);
        
        % update end_inx first, otherwise we'd be updating it using the
        % _new_ version of start_inx
        if upper_inx ~= size(x_all,2)
            end_inx = start_inx + (upper_inx - 1)*skip;
        end
        
        if lower_inx == size(x_all,2)
            start_inx = end_inx;
        else
            start_inx = start_inx + (lower_inx - 1)*skip;
        end
        
        skip = floor( (end_inx - start_inx) / (most_vecs_at_a_time-1) );
        skip = max(skip, 1);
    end
    
    xmin_low = x_all(:,lower_inx);
    xmin_high = x_all(:,upper_inx);
end

function [lower_inx, upper_inx] = binary_feasibility_search(x_all, oracle_R_unscaled, constraint)
    lower_inx = 1;
    upper_inx = size(x_all, 2);
    if oracle_R_unscaled(x_all(:,upper_inx)) <= constraint % we're already feasible
        lower_inx = upper_inx;
        return;
    end
    
    % binary search on the list of enumerate x(lambda)
    while upper_inx - lower_inx > 1
        mid_inx = floor((upper_inx + lower_inx)/2);
        x = x_all(:,mid_inx);
        if oracle_R_unscaled(x) <= constraint
            lower_inx = mid_inx;
        else
            upper_inx = mid_inx;
        end
    end
end

function x_all = greedy_sort_subsample(rho, start_inx, end_inx, skip)
    % if a=start_inx, b=end_inx, s=skip, we want to concatenate all the
    % vectors x we get with the following indices in the full ordering:
    % a, a + s, a + 2s, ..., a + ms, b
    
    if skip >= end_inx - start_inx
        m = 0;
    else
        m = floor( (end_inx - start_inx) / skip ) - 1;
    end
    num_subsampled_vecs = m + 2;
    
    % now we prepare the ordering
    rho = rho(:);
    n = length(rho);
    
    [is, js] = build_increment_ordering_mex(rho);
    x_all = fill_in_subsampled_x_all_no_mapping_mex(is, js, n, num_subsampled_vecs, start_inx, end_inx, skip);
end


function weights = compute_weights_from_regularizer(R_cell, param)
    x_lower = param.param_F.x_lower_vec;
    x_upper = param.param_F.x_upper_vec;
    k_vec = param.param_F.k_vec;
    
    weights = cell(length(R_cell),1);
    
    for ii=1:length(weights)
        R = R_cell{ii};

        k = k_vec(ii);
        weights{ii} = zeros(1,k-1);
        x_ii = interpolate(0:k-1, x_lower(ii), x_upper(ii), k_vec(ii));
        R_ii = R(x_ii); %assume each R function smart enough to be vectorized
        weights{ii} = R_ii(2:end) - R_ii(1:end-1);
    end
end