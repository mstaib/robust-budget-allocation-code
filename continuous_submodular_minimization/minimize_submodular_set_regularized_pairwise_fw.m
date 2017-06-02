function out_struct = minimize_submodular_set_regularized_pairwise_fw(F,param)
%MINIMIZE_SUBMODULAR_REGULARIZED Summary of this function goes here
%   Detailed explanation goes here

% F takes two arguments (x,param_F)
% we assume param_F is passed to us as param.param_F

param_F = param.param_F;
weights = param.weights;
n = length(weights);

H0 = F(zeros(n,1),param_F);
maxiter = param.maxiter;

verbose = parse_verbose(param);
if verbose
    fprintf('Pairwise FW - %d iterations\n', maxiter);
end

% did they supply an initial primal point rho0? (warm-start)
% (technically we are solving the dual problem)
w = parse_or_set_w0(F, param);

% initialize convex combinations; i.e. add a point for w, and set its
% weight to be one
ws{1} = w;
convex_combinations = zeros(1,maxiter+1);
convex_combinations(1) = 1;

% best primal point rho seen so far
best_rho = grad_w(w, weights);

for iter=1:maxiter
    iter_timer = tic;
    
    % compute gradient direction
    rho = grad_w(w, weights);
    
    % linear oracle
    [wbar,f,Fmin] = greedy_algorithm_set(rho,F,param.param_F);
    ws{iter+1} = wbar;
    % compute away step
    ind = find(convex_combinations(1:iter)>0);
    
    inner_prods = cellfun(@(w) dot(w, rho), ws(ind));
    [~,b] = min(inner_prods);
    b = ind(b);
    
    away_direction = w - ws{b};
    max_step_away = convex_combinations(b);
    fw_direction = wbar - w;
    direction = fw_direction + away_direction;
    max_step = max_step_away;
    
    dual_fw_pair(iter) = dot(w, rho) + regularize_sum(rho, weights);
    primal_fw_pair(iter) = f + regularize_sum(rho, weights);
    if iter > 1
        if primal_fw_pair(iter) < primal_fw_pair(iter-1)
            best_rho = rho;
        else
            primal_fw_pair(iter) = primal_fw_pair(iter-1);
        end
    end    
    
    primal_fw_pair_min(iter) = Fmin;
    fw_gap(iter) = primal_fw_pair(iter) - (dual_fw_pair(iter) + H0);
    
    step(iter) = choose_step(max_step, direction, w, weights);    
    
    convex_combination_direction = zeros(1,maxiter+1);
    convex_combination_direction(iter+1)=1;
    convex_combination_direction(b) = -1;
    convex_combinations = convex_combinations + step(iter) * convex_combination_direction;
    w = w + step(iter) * direction;
    
    iter_time(iter) = toc(iter_timer);
    
    if verbose
        fprintf('Iteration %d of %d took %0.1f seconds, duality gap %0.4e\n', iter, maxiter, iter_time(iter), fw_gap(iter));
    end
    
    if fw_gap(iter) < param.early_terminate_gap
        if verbose
            fprintf('Terminating early because of small duality gap\n');
        end
        break
    end
end

out_struct.dual_fw_pair = dual_fw_pair;
out_struct.primal_fw_pair = primal_fw_pair;
out_struct.primal_fw_pair_min = primal_fw_pair_min;
out_struct.fw_gap = fw_gap;
out_struct.step = step;
out_struct.rho = best_rho; %grad_w(ws{end}, weights); %rho;
out_struct.ws = ws;
out_struct.iter_time = iter_time;
end

function w = parse_or_set_w0(F, param)
    if isfield(param, 'rho0')
        rho = param.rho0;
        if iscell(rho)
            rho = cell2mat(rho);
        end
    else
        n = length(param.weights);
        rho = rand(n,1);
    end
    [w,~,~] = greedy_algorithm_set(rho,F,param.param_F);
end

function verbose = parse_verbose(param)
    if isfield(param, 'verbose')
        verbose = param.verbose;
    else
        verbose = true;
    end
end

function step = choose_step(max_step, direction, w, weights)
    low_step = 0;
    high_step = max_step;
    while (high_step - low_step) > 1e-6 % how to choose this tolerance?
        step1 = (2*low_step+high_step)/3;
        step2 = (low_step+2*high_step)/3;
        val1 = objective(w, step1, direction, weights);
        val2 = objective(w, step2, direction, weights);
        if val1 < val2
            low_step = step1;
        else
            high_step = step2;
        end
    end
    
    % have to take the full value high_step in case it equals max_step,
    % which corresponds to the case where we remove this element from the
    % convex_combinations set
    step = high_step;
end

function rho = grad_w(w, weights)
    rho = -w(:) ./ weights(:);
    %rho = rho(:); %so the inner product code doesn't break
end

function val = objective(w, step, direction, weights)
    wnew = w + step*direction;
    rho = grad_w(wnew, weights);
    
    val = dot(wnew, rho) + regularize_sum(rho, weights); %sum(cellfun(@(x,z) 0.5*z*sum(x.^2), rho, weights));
end

function val = regularize_sum(rho, weights)
    val = 0.5*dot(rho.^2, weights);
end