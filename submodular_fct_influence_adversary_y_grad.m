function [val, grad] = submodular_fct_influence_adversary_y_grad(x_scaled,y_mat,param)
% submodular/concave adversarial influence maximization function,

%     function [val, grad] = fct_single_advertiser(log_x_scaled,y)
%         % log_x_scaled is a matrix with the logs of the scaled values for 
%         % each x_st stored in the (s,t) coordinate
%         
%         [~,T] = size(log_x_scaled);
%         val = 0;
%         grad = zeros(size(y));
%         for t=1:T
%             % do computation for a single customer t
%             log_x_t_scaled = log_x_scaled(:,t);
%             
%             real_vars_t = real_vars(:,t);
%             yreal = y(real_vars_t);
%             log_x_t_scaled_real = log_x_t_scaled(real_vars_t);
%             log_prod = sum(yreal .* log_x_t_scaled_real);
%             
%             I_t = 1 - exp(log_prod);
% 
% %             I_t_grad = zeros(size(grad));
% %             I_t_grad(real_vars_t) = -exp(log_prod) * log_x_t_scaled_real;
%             
% %             grad = grad + I_t_grad;
%             val = val + I_t;
%             grad(real_vars_t) = grad(real_vars_t) - exp(log_prod) * log_x_t_scaled_real;
%         end
%     end

% param.k_mat is a matrix with (s,t)-th element storing the cardinality of
% the variable x(s,t)

% param.x_centers is a vector of the x(s,t) probability in the middle of
% the interval (will be zero for edges not present)

% param.delta is a scalar, the uncertainty amount for each x_{st}

% y_mat has (s,i)-th element equal to y_i(s) (i indexes the advertisers
% 1:k)

% param.alpha is a vector containing the weight for each advertiser


%x_scaled = influence_map_discrete_to_ctns(x,param);
S = length(y_mat);
y_mat_diag = spdiags(y_mat,0,S,S);

log_x_scaled = spfun(@log, x_scaled);

log_prod = y_mat_diag * log_x_scaled;
log_prod_summed = sum(log_prod);%sum along S
prods = exp(log_prod_summed); %spfun(@exp, log_prod_summed);
prods_dense = full(prods);

T = length(prods_dense);
prods_diag = spdiags(prods_dense',0,T,T);
grad_mat = log_x_scaled * prods_diag;
grad = -full(sum(grad_mat, 2));

val = T - sum(prods_dense, 2);

% which elements are actual variables and not dummy variables
%real_vars = ~(x_scaled == 0);
%x_scaled = x ./ (param.k_mat - 1) .* (x_upper - x_lower) + x_lower;

%[val, grad] = fct_single_advertiser(log_x_scaled,y_mat);

end