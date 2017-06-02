function [out_struct_max] = influence_maximization_robust(param_max, param_min, R_cell, oracle_R)

%% required inputs:

% TODO: cleanup redundancy between param_min and param_F?

% param_max:
% simplex_constraint
% max_iters
% early_terminate_gap
% verbose (optional)
%
% param_min:
% x_lower
% x_upper
% k_vec
% param_F, a struct with fields dependent on the function. For ours:
%   alpha
%   k_mat
%   x_centers
%   x_lower
%   x_upper
%   S
%   T
% rho0 (optional - for warm-starting)
% constraint
% maxiter
% early_terminate_gap
% verbose (optional)

% create our clean up object
% this way if we kill the experiment with ctrl+C, we still get the
% intermediate results
cleanupObj = onCleanup(@cleanup);


%% setup
C = param_max.simplex_constraint;

param_F = param_min.param_F;

if isfield(param_max, 'verbose')
    verbose = param_max.verbose;
else
    verbose = true;
end

y_mat_init = ones(param_F.S,1);
y_mat_init = proj_feas(y_mat_init);

% initialize with the oblivious solution
%[ y_mat, Fval, out, opts ] = solve_influence_fixed_x(zeros(size(param_min.param_F.S,param_min.param_F.T)), y_mat_init, param_F, C);
[ y_mat, Fval, out, opts ] = solve_influence_fixed_x(zeros(size(param_min.k_vec(:))), y_mat_init, param_F, C);



%% actual outer loop

xmin_ave_unscaled = zeros(size(param_min.k_vec(:)));

for kk=1:param_max.maxiters
    this_iter = tic;
    F = @(x,param_F) submodular_fct_influence_adversary_vec_wrapper(x,y_mat,param_F);
    param_F.y_mat = y_mat;
    param_min.param_F = param_F;

    param_min.y_mat = y_mat; %always store the actual y values so minimize_submdoular_constrained can figure out if it's DR-submodular
    [xmin_unscaled,Fmin,out_struct,certified_suboptimality_gap] = minimize_submodular_constrained(F,R_cell,oracle_R,param_min);
    param_min.weights = out_struct.weights;
    
%     %% the following block (until the next %%) is for testing the gradient-based solution
%     [xmin_scaled, out_struct] = gradient_descent_constrained(R_cell,param_min);
%     out_struct.rho = [];
%     certified_suboptimality_gap = [];
%     xmin_unscaled = discretize(xmin_scaled, param_F.x_lower_vec, param_F.x_upper_vec, param_F.k_vec);
%     %%
    
    constrained_certified_suboptimality_gap{kk} = certified_suboptimality_gap;
    xmin_unscaled = xmin_unscaled(:);
    param_min.rho0 = out_struct.rho;
    rhos{kk} = out_struct.rho;
    %param_min.w0 = out_struct.ws{end}; % if we are using FW on the dual
    
    % storing out_struct_min takes too much memory for large instances
    %out_struct_max.out_struct_min{kk} = out_struct;
    
    %xmin = interpolate(xmin_unscaled, param_min.x_lower, param_min.x_upper, param_min.k_vec);
    %xmin_ave = (xmin_ave * (kk - 1) + xmin) / kk;
    %xmin_ave_unscaled = discretize(xmin_ave, param_min.x_lower, param_min.x_upper, param_min.k_vec);
    xmin_ave_unscaled = (xmin_ave_unscaled * (kk - 1) + xmin_unscaled) / kk;
    
    Fval_hist(kk) = submodular_fct_influence_adversary_vec_wrapper(xmin_unscaled,y_mat,param_F);
    
    % how to extract the gradient wrt y?
    [~, y_grad] = submodular_fct_influence_adversary_y_grad_vec_wrapper(xmin_unscaled,y_mat,param_F);
    
    % here we get a duality gap by maximizing y for the fixed (current) x
    % we multiplied the function by -1 so tfocs does maximization for us
    [ y_mat_fixed_x, Fval, out, opts ] = solve_influence_fixed_x(xmin_ave_unscaled, y_mat, param_F, C);
    
    dual_bound(kk) = Fval;

    if kk > 1
        % dual bound should never increase
        dual_bound(kk) = min(dual_bound(kk), dual_bound(kk-1));
    end
    dual_gap(kk) = dual_bound(kk) - Fval_hist(kk);
    least_dual_gap(kk) = dual_bound(kk) - max(Fval_hist(kk));
    stepsize(kk) = least_dual_gap(kk) / sum(sum(y_grad.^2));
    
    % how do we want to choose step size? what does the dual problem look
    % like and can we get a gap?
    y_mat = y_mat + stepsize(kk) * y_grad;
    y_mat = proj_feas(y_mat);
    
    y_mat_hist{kk} = y_mat;
    
    iter_time(kk) = toc(this_iter);
    
    if verbose
        fprintf('Duality gap at iteration %d: %0.4e, ', kk, dual_gap(kk));
        fprintf('Took %0.1f seconds, ', iter_time(kk));
        fprintf('%d inner iters, ', length(out_struct.iter_time));
        if isfield(out_struct, 'multiplier_time')
            fprintf('%0.1f seconds for multiplier\n', out_struct.multiplier_time);
        else
            fprintf('\n');
        end
        %fprintf('stepsize: %0.4e, ', stepsize(kk));
        %fprintf('grad norm: %0.4e\n', sum(sum(y_grad.^2)));
    end
    if dual_gap(kk) < param_max.early_terminate_gap
        if verbose
            fprintf('Terminating early because of small duality gap\n');
        end
        break
    end
    if kk > 1 && dual_gap(kk) == dual_gap(kk-1)
        if verbose
            fprintf('Terminating early because made no progress on duality gap\n');
        end
        break
    end
end

% we duplicate this code because otherwise we lose the last values
Fval_hist(kk+1) = submodular_fct_influence_adversary_vec_wrapper(xmin_unscaled,y_mat,param_F);

% here we get a duality gap by maximizing y for the fixed (current) x
% we multiplied the function by -1 so tfocs does maximization for us
% NEVERMIND: xmin_ave_unscaled has not changed since we last computed this
%[ ~, Fval, ~, ~ ] = solve_influence_fixed_x(xmin_ave_unscaled, y_mat, param_F, C);
%dual_bound(kk+1) = Fval;
dual_bound(kk+1) = dual_bound(kk);
% % dual bound should never increase
% dual_bound(kk+1) = min(dual_bound(kk+1), dual_bound(kk));
dual_gap(kk+1) = dual_bound(kk+1) - Fval_hist(kk+1);


%% interpretation
% continuous version of xmin
xmin = interpolate(xmin_unscaled, param_F.x_lower_vec, param_F.x_upper_vec, param_min.k_vec);

% how badly would we have done assuming no adversarial perturbations?
[ y_mat_oblivious, Fval, out, opts ] = solve_influence_fixed_x(zeros(size(xmin_unscaled)), y_mat, param_F, C);

% we need to optimize over the xs for this particular y
F = @(x,param_F) submodular_fct_influence_adversary_vec_wrapper(x,y_mat_oblivious,param_F);
% 
% if isfield(param_min, 'w0')
%     param_min = rmfield(param_min, 'w0');
% end
% if isfield(param_min, 'rho0')
%     param_min = rmfield(param_min, 'rho0');
% end
% [xmin_oblivious_unscaled,Fmin_oblivious,out_struct] = minimize_continuous_submodular_constrained(F,param_min);
% oblivious_gap = max(Fval_hist) - Fmin_oblivious;
% 
% % how do they both do in the case with no adversary?data
% robust_no_adversary_val = submodular_fct_influence_adversary_vec_wrapper(zeros(size(xmin_unscaled)),y_mat,param_F);
% oblivious_no_adversary_val = submodular_fct_influence_adversary_vec_wrapper(zeros(size(xmin_unscaled)),y_mat_oblivious,param_F);


%% store all we want
out_struct_max.Fval_hist = Fval_hist;
out_struct_max.dual_bound = dual_bound;
out_struct_max.dual_gap = dual_gap;
out_struct_max.x_dual_certificate_unscaled = xmin_ave_unscaled;
out_struct_max.stepsize = stepsize;
out_struct_max.iter_time = iter_time;
out_struct_max.y_mat_hist = y_mat_hist;
out_struct_max.xmin = xmin;
out_struct_max.xmin_unscaled = xmin_unscaled;
% out_struct_max.xmin_oblivious_unscaled = xmin_oblivious_unscaled;
% out_struct_max.Fmin_oblivious = Fmin_oblivious;
out_struct_max.y_mat_oblivious = y_mat_oblivious;
% out_struct_max.oblivious_gap = oblivious_gap;
% out_struct_max.rel_oblivious_gap = oblivious_gap / Fmin_oblivious;
out_struct_max.constrained_certified_suboptimality_gap = constrained_certified_suboptimality_gap;
out_struct_max.rhos = rhos;

% out_struct_max.robust_no_adversary_val = robust_no_adversary_val;
% out_struct_max.oblivious_no_adversary_val = oblivious_no_adversary_val;

%% functions

% project onto feasible set
function proj_y_out = proj_feas(y_mat)   
    try
        proj = proj_simplex(C, true, false);
        [~, proj_y_out] = proj(y_mat, 1);
    catch ME
        fprintf('Caught exception in projection step\n');
        throw(ME);
    end
        
end

function cleanup()    
%     filename = [datestr(now,'yyyy-mm-dd_HHMMSS') '.mat'];
%     fprintf('saving data so far to file %s\n', filename);
%     save(filename,...
%         'Fval_hist',...
%         'dual_bound',...
%         'dual_gap',...
%         'xmin_ave_unscaled',...
%         'stepsize',...
%         'iter_time',...
%         'y_mat_hist',...
%         'xmin_unscaled',...
%         'constrained_certified_suboptimality_gap',...
%         'rhos');
end

end