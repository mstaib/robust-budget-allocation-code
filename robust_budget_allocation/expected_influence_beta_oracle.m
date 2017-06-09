function [ fy, gy ] = expected_influence_beta_oracle(y_mat,param)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%   TFOCS style oracle -- fy is value, gy is gradient

alphas = param.alphas;
betas = param.betas;
edges = param.real_edges;

[fy, gy] = fct_single_advertiser(y_mat,alphas,betas,edges);

end

function [ fy, gy ] = fct_single_advertiser(y_vec,alphas,betas,edges)
    [S,T] = size(alphas);

    fy = 0;
    gy = zeros(size(y_vec));
    
    for t=1:T
        edges_t = edges(:,t);
        if ~any(edges_t)
            continue;
        end
        
        alphas_t = alphas(edges_t,t);
        betas_t = betas(edges_t,t);
        y_t = y_vec(edges_t);
        
        %vals_t = beta(alphas_t + y_t, betas_t) ./ beta(alphas_t,betas_t);
        log_vals_t = betaln(alphas_t + y_t, betas_t) - betaln(alphas_t, betas_t);
        vals_t = exp(log_vals_t);
        
        fy = fy + prod(vals_t);
        gy(edges_t) = gy(edges_t) + vals_t .* (psi(alphas(edges_t,t) + y_vec(edges_t)) - psi(alphas(edges_t,t) + y_vec(edges_t) + betas(edges_t,t)));
    end
    
    fy = T - fy;
    gy = -gy;
end