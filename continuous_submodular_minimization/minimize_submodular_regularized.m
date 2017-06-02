function out_struct = minimize_submodular_regularized(F,param)
%MINIMIZE_SUBMODULAR_REGULARIZED Summary of this function goes here
%   Detailed explanation goes here

% F takes two arguments (x,param_F)
% we assume param_F is passed to us as param.param_F
%
% later we will generalize to other regularizers, but first we will start
% with t * sum x_i, i.e. a_{i x_i}(t) = 1/2*t^2

param_F = param.param_F;

weights = param.weights;
%min_weight = min(weights);
%weights = weights / min_weight;
weights_cell = num2cell(weights);

k_vec = param.k_vec;
n = length(k_vec);

%% subgradient descent on [0,1]
H0 = F(zeros(1,n),param_F);
maxiter = param.maxiter;

% did they supply an initial point rho0? (warm-start)
if isfield(param, 'rho0') 
    rho = param.rho0;
else
    % random initialization (with varying k now)
    rho = cell(n,1);
    for ii=1:n
        rho{ii} = fliplr(cumsum(rand(1,k_vec(ii)-1),2));
    end
    
    min_rho = min(cellfun(@min, rho));
    rho = cellfun(@(x) x-min_rho, rho,'UniformOutput',false);
    max_rho = max(cellfun(@max, rho));
    rho = cellfun(@(x) x/max_rho, rho,'UniformOutput',false);
end

if isfield(param, 'verbose')
    verbose = param.verbose;
else
    verbose = true;
end

w_ave = 0;

if verbose
    fprintf('Subgradient - %d iterations\n', maxiter);
end

strong_convex_parm = min(weights);

for iter=1:maxiter
    iter_timer = tic;
    
    [w,f,Fmin] = greedy_algorithm(rho,F,param.param_F);
    
    % should really rename this because w is a little ambiguous now
    subgrad = cellfun(@(x,y,z) x + z*y, w, rho, weights_cell, ...
        'UniformOutput', false);
    
    if iter==1
        w_ave = cellfun(@(x) zeros(size(x)), w, 'UniformOutput', false);
    end
    
    % computing a dual candidate
    w_ave = cellfun(@(x_ave, x) (x_ave * (iter - 1) + x)/iter, w_ave, w, 'UniformOutput', false);
    
    regularize_val = sum(cellfun(@(x,z) 0.5*z*sum(x.^2), rho, weights_cell));
    primal_subgradient(iter)=f + regularize_val;
    primal_subgradient_min(iter) = Fmin; % + regularize_val; Bach's original code did not add regularize_val back in
    
    dual_subgradient(iter) = sum(cellfun(@(x,z) sum(-0.5*z^(-1)*(-x).^2), w_ave, weights_cell));
        
    % subgradient step (Polyak rule)
    norm_subgrad_squared = sum( cellfun(@(x) sum(x.^2), subgrad) );
    
    % TODO: is this actually how to calculate the duality gap?????
    dual_gap(iter) = primal_subgradient(iter) - (max(dual_subgradient) + H0);
    if dual_gap(iter) < 0
        dual_gap(iter) = 0;
    end

    %stepsize(iter) = dual_gap(iter) / norm_subgrad_squared; % polyak
    stepsize(iter) = 2 / (strong_convex_parm * (iter + 1)); % https://arxiv.org/pdf/1212.2002v2.pdf
    
    rho = cellfun(@(x,y) x - stepsize(iter) * y, rho, subgrad, 'UniformOutput', false);
        
    % orthogonal projection onto the feasible (monotone) set for rho
    for i=1:n
        rho{i} = -pav(-min(1,max(rho{i},0)));
    end
    
    if verbose
        fprintf('Iteration %d of %d took %0.1f seconds, duality gap %0.4e\n', iter, maxiter, toc(iter_timer), dual_gap(iter));
    end
    
    if dual_gap(iter) < param.early_terminate_gap
        if verbose
            fprintf('Terminating early because of small duality gap\n');
        end
        break
    end
    
    if iter > 1 && abs(dual_gap(iter) - dual_gap(iter-1)) < 1e-9
        if verbose
            fprintf('Terminating early because we made no progress from the last iteration\n');
            fprintf('Dual gap was %0.4e\n', dual_gap(iter));
            fprintf('This likely means that the optimal dual var w lies in W(H) and not in B(H)\n');
        end
        break
    end
end

out_struct.primal_subgradient = primal_subgradient;
out_struct.primal_subgradient_min = primal_subgradient_min;
out_struct.dual_subgradient = dual_subgradient;
out_struct.dual_gap = dual_gap;
out_struct.stepsize = stepsize;
out_struct.rho = rho;
end

