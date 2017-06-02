function val = submodular_fct_influence_adversary_vec_wrapper_pmatlab(x_vec,y_mat,param)
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

var_edges_inx = param.var_edges_inx;
save('pmatlab_script_input', 'num_vecs', 'log_x_centers_sparse', 'y_mat_diag', 'var_edges_inx');

scriptName = 'submodular_fct_influence_adversary_vec_wrapper_pmatlab_script';
np=64;
eval(pRUN(scriptName,np,'grid&'));

while LLGrid_myjobs
    pause(0.1)
end

val = zeros(num_vecs,1);
for ii=0:np-1
    load(sprintf('pmatlab_script_output_%d', ii));
    val(myIndices) = myVal;
end
    
end