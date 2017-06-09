function val = submodular_fct_influence_adversary_vec_wrapper_update_based(xold, is, js, param)
% wrapper for submodular_fct_influence_adversary which converts x_vec to
% a matrix as needed

%curr_x = xold;

%N = length(is);
%val = zeros(N+1, 1);

% fill in the first one to get us started
mult = (param.x_upper_vec - param.x_lower_vec) ./ param.k_vec;
xold_var_edges = mult .* xold + param.x_lower_vec;
curr_x = xold_var_edges;
log_x = spfun(@log, param.x_centers_sparse);
y_mat_diag = spdiags(param.y_mat,0,param.S,param.S);

[val_init, log_prods] = evaluate_from_scratch(curr_x, param, log_x, y_mat_diag);
%val(1) = val_init;

% log_x(param.var_edges_inx) = log(xold_var_edges);
% 
% log_prod = y_mat_diag * log_x;
% log_prod_summed = sum(log_prod);
% log_prods = full(log_prod_summed);
% val(1) = param.T - sum(exp(log_prods), 2);

[S_affected_vec, T_affected_vec] = ind2sub([param.S, param.T], param.var_edges_inx);
val = incremental_differences_loop_mex(curr_x, is, js, val_init, log_prods, mult, param.x_lower_vec, param.y_mat, S_affected_vec, T_affected_vec);
    
end

function [val] = incremental_differences_loop(curr_x, is, js, val_init, log_prods, mult, x_lower_vec, y_mat, S_vec, T_vec)
    N = length(is);
    val = zeros(N+1, 1);
    val(1) = val_init;
    
    for kk=1:N
        i = is(kk); j = js(kk);
        old_val = curr_x(i);
        new_val = mult(i) * j + x_lower_vec(i);
        curr_x(i) = new_val;

        %[S_affected, T_affected] = ind2sub([param.S, param.T], param.var_edges_inx(i));
        S_affected = S_vec(i); T_affected = T_vec(i);
        log_diff = y_mat(S_affected) * (log(new_val) - log(old_val));
        log_prods_T_new = log_prods(T_affected) + log_diff;
        exp_diff = exp(log_prods_T_new) - exp(log_prods(T_affected));

        log_prods(T_affected) = log_prods_T_new; 

        val(kk+1) = val(kk) - exp_diff;
    end
end

function [val, log_prods] = evaluate_from_scratch(x_var_edges, param, log_x, y_mat_diag)
    log_x(param.var_edges_inx) = log(x_var_edges);
    
    log_prod = y_mat_diag * log_x;
    log_prod_summed = sum(log_prod);
    log_prods = full(log_prod_summed);
    val = param.T - sum(exp(log_prods), 2);
end