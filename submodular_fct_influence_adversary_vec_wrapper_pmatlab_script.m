PARALLEL=1;

% variables that should be in scope:
% num_vecs
% log_x_centers_sparse
% y_mat_diag
% (param.)var_edges_inx
load 'pmatlab_script_input';

if PARALLEL
    myIndices = global_ind(zeros(num_vecs, 1, map([Np 1],'b',0:Np-1)));
    
    myVal = zeros(length(myIndices), 1);
    for ii=1:length(myIndices)
        inx = myIndices(ii);
        log_x = log_x_centers_sparse;
        log_x(var_edges_inx) = log_x_var_edges(:,inx);
        myVal(ii) = submodular_fct_influence_adversary_sparse(log_x,y_mat_diag);
    end
    
    save(sprintf('pmatlab_script_output_%d', Np), 'myVal', 'myIndices');
else
    
end