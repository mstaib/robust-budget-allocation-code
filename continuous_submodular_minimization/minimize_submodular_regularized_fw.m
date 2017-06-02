function out_struct = minimize_submodular_regularized_fw(F,param)
%MINIMIZE_SUBMODULAR_REGULARIZED Summary of this function goes here
%   Detailed explanation goes here

% F takes two arguments (x,param_F)
% we assume param_F is passed to us as param.param_F
%
% later we will generalize to other regularizers, but first we will start
% with t * sum x_i, i.e. a_{i x_i}(t) = 1/2*t^2

param_F = param.param_F;

weights = param.weights;
weights_cell = num2cell(weights);

k_vec = param.k_vec;
n = length(k_vec);

%% subgradient descent on [0,1]
H0 = F(zeros(n,1),param_F);
maxiter = param.maxiter;

if isfield(param, 'verbose')
    verbose = param.verbose;
else
    verbose = true;
end

if verbose
    fprintf('(regular) FW - %d iterations\n', maxiter);
end

% did they supply an initial point w0? (warm-start)
if isfield(param, 'rho0')
    rho = param.rho0;
else
    rho = cell(n,1);
    for ii=1:n
        rho{ii} = flipud(cumsum(rand(k_vec(ii)-1,1),1));
    end
    min_rho = min(cellfun(@min, rho));
    rho = cellfun(@(x) x-min_rho, rho,'UniformOutput',false);
    max_rho = max(cellfun(@max, rho));
    rho = cellfun(@(x) x/max_rho, rho,'UniformOutput',false);
end
[w,~,~] = greedy_algorithm(rho,F,param.param_F);

ws{1} = w;

for iter=1:maxiter
    iter_timer = tic;
    
    % compute gradient direction
    rho = grad_w(w, weights);
    
    % linear oracle
    [wbar,f,Fmin] = greedy_algorithm(rho,F,param.param_F);
    ws{iter+1} = wbar;
    % compute away step
    direction = wbar;
    
    max_step = 1;
    
    dual_fw_pair(iter) = cell_innerprod(w, rho) + regularize_sum(rho, weights);
    primal_fw_pair(iter) = f + regularize_sum(rho, weights);
    primal_fw_pair_min(iter) = Fmin;
    fw_gap(iter) = primal_fw_pair(iter) - dual_fw_pair(iter) - H0;
    %dual_fw_pair_min(iter)  
    
    % line search
%     aa = cell_innerprod(rho, direction);
%     bb = sum( cellfun(@(x) sum(x.^2), direction) );
%     step(iter) = min(max_step,max(aa/bb,0));
    
    low_step = 0;
    high_step = max_step;
    low_val = fw_dual_objective_custom(w, low_step, direction, weights);
    high_val = fw_dual_objective_custom(w, high_step, direction, weights);
    while (high_step - low_step) > 1e-6 * max_step % how to choose this?
        step1 = (2*low_step+high_step)/3;
        step2 = (low_step+2*high_step)/3;
        val1 = fw_dual_objective_custom(w, step1, direction, weights);
        val2 = fw_dual_objective_custom(w, step2, direction, weights);
        if val1 < val2
            low_step = step1;
        else
            high_step = step2;
        end
    end
    
    step(iter) = (high_step + low_step)/2;
    %step(iter) = 2/(iter+1);
    
    w = cellfun(@(x,y) (1-step(iter)) * x + step(iter) * y, w, direction, 'UniformOutput', false);
    
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
out_struct.rho = rho;
out_struct.ws = ws;
out_struct.iter_time = iter_time;
end

function val = cell_innerprod(w, rho)
    val = sum(cellfun(@(wi,rhoi) wi(:)'*rhoi(:), w, rho));
end

function rho = grad_w(w, weights)
    rho = cell(sort(size(w),'descend'));
    for i=1:length(w)
        %for when we generalize weights to each coordinate
        rho{i} = -pav_custom(weights{i}.^(-1).*w{i},weights{i});
        %rho{i} = -pav(weights(i)^(-1)*w{i},repmat(weights(i),1,length(w{i})));
    end
    %rho = rho(:); %so the inner product code doesn't break
end

function val = objective(w, step, direction, weights)
    wnew = cellfun(@(x,y) x + step*y, w, direction, 'UniformOutput', false);
    rho = grad_w(wnew, weights);
    
    val = cell_innerprod(wnew, rho) + regularize_sum(rho, weights); %sum(cellfun(@(x,z) 0.5*z*sum(x.^2), rho, weights));
end

function val = regularize_sum(rho, weights)
    val = 0;
    for kk=1:length(weights)
        val = val + 0.5*sum(rho{kk}.^2 .* weights{kk});
    end
end