function [f, g] = submodular_fct_influence_adversary_y_grad_vec_wrapper(x_vec,y_mat,param)
% wrapper for submodular_fct_influence_adversary which converts x_vec to
% a matrix as needed

x = param.x_centers_sparse;
x(param.var_edges) = interpolate(x_vec, param.x_lower_vec, param.x_upper_vec, param.k_vec);

[f, g] = submodular_fct_influence_adversary_y_grad(x,y_mat,param);        

end

