function [w,f,Fmin] = greedy_algorithm(rho,F,param_F)
% greedy algorithm (assumes all variables have the same cardinality)

% make sure rho is n x 1
rho = rho(:);
n = length(rho);

%[is, js] = build_increment_ordering(rho);
[is, js] = build_increment_ordering_mex(rho);
%num_total_rhos = length(is);

rho_lengths = cellfun(@length, rho);

% now go through all elements
xold = zeros(n,1);
H0 = F(xold,param_F);

% F_vals_all(1) is just F(xold,param_F)
F_vals_all = submodular_fct_influence_adversary_vec_wrapper_update_based(xold, is, js, param_F);

[w, Fmin] = build_w_from_F_vals_mex(F_vals_all, is, js, rho_lengths);

% % TODO: mexify the below
% Fmin = Fold;
% for i=1:num_total_rhos
%     Fnew = F_vals_all(i+1);
%     if (Fnew<Fmin), Fmin = Fnew; end
%     w_i = w{is(i)};
%     w_i(js(i)) = Fnew - Fold;
%     w{is(i)} = w_i;
%     Fold = Fnew;
% end

% f is affine in the rho; the variable f is the actual function value, and
% the vector w is a subgradient

%f = sum(cellfun(@(x,y) x(:)'*y(:), w, rho)) + H0;
f = cell_innerprod_mex(w, rho) + H0;
end

function [w, Fmin] = build_w_from_F_vals(F_vals_all, is, js, rho_lengths)
    n = length(rho_lengths);

    w = cell(n, 1);
    for ii=1:n
        w{ii} = zeros(rho_lengths(ii), 1);
    end

    Fold = F_vals_all(1);

    Fmin = Fold;
    for i=1:length(is)
        Fnew = F_vals_all(i+1);
        if (Fnew<Fmin), Fmin = Fnew; end
        w_i = w{is(i)};
        w_i(js(i)) = Fnew - Fold;
        w{is(i)} = w_i;
        Fold = Fnew;
    end
end