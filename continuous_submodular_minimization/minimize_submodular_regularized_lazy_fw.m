function out_struct = minimize_submodular_regularized_lazy_fw(F,param)
%MINIMIZE_SUBMODULAR_REGULARIZED_LAZY_FW Summary of this function goes here
%   Detailed explanation goes here

%% setup including generating initial feasible point w
param_F = param.param_F;

weights = param.weights;
weights_cell = num2cell(weights);

k_vec = param.k_vec;
n = length(k_vec);

H0 = F(zeros(n,1),param_F);
maxiter = param.maxiter;

if isfield(param, 'verbose')
    verbose = param.verbose;
else
    verbose = true;
end

if verbose
    fprintf('(lazy) FW - %d iterations\n', maxiter);
end

% did they supply an initial point w0? (warm-start)
if isfield(param, 'rho0') && length(param.rho0) == length(param.param_F.var_edges_inx)
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

% LPSep oracle
% we expect c = gradient
cached_vertices = {};
function y = lp_sep(c, x, Phi, K)
    num_cached = length(cached_vertices);
    
    if num_cached >= 1
        for kk=1:num_cached
            y_k = cached_vertices{kk};
            x_minus_y_k = cellfun(@(a,b) a - b, x, y_k, 'UniformOutput', false);
            if cell_innerprod(x_minus_y_k, c) < -Phi / K
                y = y_k;
                return
            end
        end
    end
    
    [y,~,~] = greedy_algorithm(c,F,param.param_F);
    cached_vertices{num_cached + 1} = y;
end

% initialize Phi
rho = grad_w(w, weights);
[z,~,~] = greedy_algorithm(rho,F,param.param_F);
w_minus_z = cellfun(@(a,b) a - b, w, z, 'UniformOutput', false);
Phi = -cell_innerprod(w_minus_z, rho);

K = 1.01;

for iter=1:maxiter
    iter_timer = tic;
        
    % compute gradient direction of this block
    rho = grad_w(w, weights);
    
    direction = lp_sep(rho, w, Phi, K);
    
    
%     dual_fw_pair(iter) = cell_innerprod(w, rho) + regularize_sum(rho, weights);
%     primal_fw_pair(iter) = f + regularize_sum(rho, weights);
%     primal_fw_pair_min(iter) = Fmin;
    %primal_fw_pair(iter) - dual_fw_pair(iter) - H0;
    %dual_fw_pair_min(iter)  
    
    % line search
%     aa = cell_innerprod(rho, direction);
%     bb = sum( cellfun(@(x) sum(x.^2), direction) );
%     step(iter) = min(max_step,max(aa/bb,0));
    
    max_step = 1;
    
    low_step = 0;
    high_step = max_step;
    low_val = fw_dual_objective_custom(w, low_step, direction, weights);
    high_val = fw_dual_objective_custom(w, high_step, direction, weights);
    while (high_step - low_step) > 1e-6 % how to choose this?
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
    
    w_new = cellfun(@(x,y) (1-step(iter)) * x + step(iter) * y, w, direction, 'UniformOutput', false);
    
    % now let's see if we can shrink Phi
    w_minus_direction = cellfun(@(a,b) a - b, w, direction, 'UniformOutput', false);
    innerprod = -cell_innerprod(w_minus_direction, rho);
    if innerprod <= Phi / K
        Phi = innerprod / 2;
    end
    
    % finally update the new guy
    w = w_new;
    
    % compute a duality gap bound
    fw_gap(iter) = 2*Phi; 
    
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

% out_struct.dual_fw_pair = dual_fw_pair;
% out_struct.primal_fw_pair = primal_fw_pair;
% out_struct.primal_fw_pair_min = primal_fw_pair_min;
out_struct.fw_gap = fw_gap;
out_struct.step = step;
out_struct.rho = rho;
out_struct.iter_time = iter_time;
end

function val = cell_innerprod(w, rho)
    val = sum(cellfun(@(wi,rhoi) wi(:)'*rhoi(:), w, rho));
end

function rho_i = grad_w_block(w_i, weights_i)
    rho_i = -pav_mex(weights_i.^(-1) .* w_i, weights_i);
end

function rho = grad_w(w, weights)
    rho = cell(sort(size(w),'descend'));
    for i=1:length(w)
        rho{i} = grad_w_block(w{i}, weights{i});
    end
end

function val = objective(w, step, direction, weights)
    wnew = cellfun(@(x,y) (1-step)*x + step*y, w, direction, 'UniformOutput', false);
    rho = grad_w(wnew, weights);
    
    val = cell_innerprod(wnew, rho) + regularize_sum(rho, weights); %sum(cellfun(@(x,z) 0.5*z*sum(x.^2), rho, weights));
end

function val = regularize_sum(rho, weights)
    val = 0;
    for kk=1:length(weights)
        val = val + 0.5*sum(rho{kk}.^2 .* weights{kk});
    end
end