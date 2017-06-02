function [xmin,Fmin,out_struct] = minimize_submodular(F,param)
%MINIMIZE_SUBMODULAR Summary of this function goes here
%   Detailed explanation goes here

% F takes two arguments (x,param_F)
% we assume param_F is passed to us as param.param_F
param_F = param.param_F;

k_vec = param.k_vec;
n = length(k_vec);

%% subgradient descent on [0,1]
H0 = F(zeros(1,n),param_F);
maxiter = param.maxiter;

% random initialization (with varying k now)
rho = cell(n,1);
for ii=1:n
    rho{ii} = fliplr(cumsum(rand(1,k_vec(ii)-1),2));
end
%rho = fliplr(cumsum(rand(n,k-1),2));
min_rho = min(cellfun(@min, rho));
rho = cellfun(@(x) x-min_rho, rho,'UniformOutput',false);
max_rho = max(cellfun(@max, rho));
rho = cellfun(@(x) x/max_rho, rho,'UniformOutput',false);

w_ave = 0;

fprintf('Subgradient - %d iterations\n', maxiter);

for iter=1:maxiter
    iter_timer = tic;
    
    %if mod(iter,10)==0, fprintf('%d ', iter); end
    [w,f,Fmin] = greedy_algorithm(rho,F,param.param_F);
    
    if iter==1
        w_ave = cellfun(@(x) zeros(size(x)), w, 'UniformOutput', false);
    end
    
    % computing a dual candidate
    w_ave = cellfun(@(x_ave, x) (x_ave * (iter - 1) + x)/iter, w_ave, w, 'UniformOutput', false);
    
    primal_subgradient(iter)=f;
    primal_subgradient_min(iter) = Fmin;
    
    cumsums = cellfun(@cumsum, w_ave, 'UniformOutput', false);
    dual_subgradient(iter) = sum( min(cellfun(@min,cumsums),0) );
    
    % subgradient step (Polyak rule)
    norm_w_squared = sum( cellfun(@(x) sum(x.^2),w) );
    stepsize = ( f - max(dual_subgradient) ) / norm_w_squared;
    rho = cellfun(@(x,y) x - stepsize * y, rho, w, 'UniformOutput', false);
        
    % orthogonal projection onto the feasible (monotone) set for rho
    for i=1:n,
        rho{i} = -pav(-min(1,max(rho{i},0)));
    end
    
    %if iter == 1
    fprintf('Done with iteration %d of %d in %0.1f seconds\n', iter, maxiter, toc(iter_timer));
    %end
end
rho_subgradient = rho;


[xmin,Fmin] = theta_minimizer(rho,F,param_F);

% % copied more or less from submodular_fct_influence_adversary + some code
% % from the wrapper (to make our variable a vector)
% function x_scaled = map_discrete_to_ctns(x_vec, param_F)
%     x = zeros(param_F.S,param_F.T);
%     x(param_F.var_indices) = x_vec;
%     
%     dummy = param_F.k_mat == 0;
%     x_scaled = zeros(size(x));
%     x_scaled(dummy) = param_F.x_centers(dummy);
%     x_scaled(~dummy) = x(~dummy) ./ (param_F.k_mat(~dummy) - 1) ...
%         .* (param_F.x_upper(~dummy) - param_F.x_lower(~dummy)) ...
%         + param_F.x_lower(~dummy);
%     x_scaled = reshape(x_scaled, size(x));
% end

xmin = interpolate(xmin, param.x_lower, param.x_upper, param.k_vec);
%xmin = influence_map_discrete_to_ctns(xmin, param.param_F);

out_struct.primal_subgradient = primal_subgradient;
out_struct.primal_subgradient_min = primal_subgradient_min;
out_struct.dual_subgradient = dual_subgradient;
end

