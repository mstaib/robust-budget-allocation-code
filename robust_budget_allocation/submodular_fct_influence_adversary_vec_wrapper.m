function val = submodular_fct_influence_adversary_vec_wrapper(x_vec,y_mat,param)
% wrapper for submodular_fct_influence_adversary which converts x_vec to
% a matrix as needed

[~, num_vecs] = size(x_vec);

mult = (param.x_upper_vec - param.x_lower_vec) ./ param.k_vec;
x_var_edges = mult .* x_vec + param.x_lower_vec;

% how many channels
S = length(y_mat);
y_mat_diag = spdiags(y_mat,0,S,S);

log_x_centers_sparse = spfun(@log, param.x_centers_sparse);
log_x_var_edges = log(x_var_edges);

val = zeros(num_vecs,1);
for ii=1:num_vecs
    log_x = log_x_centers_sparse;
    log_x(param.var_edges_inx) = log_x_var_edges(:,ii);
    val(ii) = submodular_fct_influence_adversary_sparse(log_x,y_mat_diag);
end
    
end