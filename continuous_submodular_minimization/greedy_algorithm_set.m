function [w,f,Fmin] = greedy_algorithm_set(rho,F,param_F)
% greedy algorithm (assumes all variables have the same cardinality)

% make sure rho is n x 1
rho = rho(:);
n = length(rho);

w = zeros(size(rho));

% first order all rhos (does preserve the ordering within rows if equal
% values)
%all_rhos = [rho{:}];
%num_total_rhos = length(all_rhos);
[~, s] = sort(rho, 'descend');

% now go through all elements
xold = zeros(n,1);
Fold = F(xold,param_F);
H0 = Fold;

chunk_total_size = 1e8;
chunk = floor(chunk_total_size / n);
chunk = max(1,chunk);


% we pregenerate batches of x values so we can compute F of all of them in
% parallel

F_vals = zeros(n,1);

Fmin = Fold;
num_chunks = ceil(n / chunk);
for kk=1:num_chunks
    elem_range = (1+(kk-1)*chunk):min(n, kk*chunk);

    x_all = build_all_x_vectors(xold, s(elem_range));
    F_vals(elem_range) = F(x_all, param_F);

    for i=1:length(elem_range)
        xnew = x_all(:,i); Fnew = F_vals(elem_range(i));
        if (Fnew<Fmin), Fmin = Fnew; end
        w(s(elem_range(i))) = Fnew - Fold;
        xold = xnew;
        Fold = Fnew;
    end
end


% f is affine in the rho; the variable f is the actual function value, and
% the vector w is a subgradient
f = dot(w,rho) + H0;
end