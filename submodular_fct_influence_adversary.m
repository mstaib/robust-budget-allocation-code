function g = submodular_fct_influence_adversary(x_scaled,y_mat,param)
%#codegen
% submodular/concave adversarial influence maximization function,

% x_scaled is an S x T matrix already scaled to continuous values and with
% holes replaced with x_centers

% y_mat has (s,i)-th element equal to y_i(s) (i indexes the advertisers
% 1:k)

% alpha is a vector containing the weight for each advertiser


%x_scaled = map_discrete_to_ctns(x,param);
log_x_scaled = log(x_scaled);

% which elements are actual variables and not dummy variables
real_vars = ~(x_scaled == 0);

% number of advertisers
k = length(param.alpha);

g = 0;
for ii=1:k
    g = g + param.alpha(ii) * fct_single_advertiser(log_x_scaled,y_mat(:,ii),real_vars);
end

end

function f = fct_single_advertiser(log_x_scaled,y,real_vars)
%#codegen
    % log_x_scaled is a matrix with the logs of the scaled values for 
    % each x_st stored in the (s,t) coordinate

    [~,T] = size(log_x_scaled);
    f = 0;
    for t=1:T
        % do computation for a single customer t
        log_x_t_scaled = log_x_scaled(:,t);

        real_vars_t = real_vars(:,t);
        yreal = y(real_vars_t);
        log_x_t_scaled_real = log_x_t_scaled(real_vars_t);
        log_prod = sum(yreal .* log_x_t_scaled_real);

        I_t = 1 - exp(log_prod);

        f = f + I_t;
    end
end