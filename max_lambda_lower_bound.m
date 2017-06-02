function [ lambdas, vals ] = max_lambda_lower_bound(F, param, R_cell, rho, B )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
n = length(rho);

all_rhos = vertcat(rho{:});

lambdas = all_rhos;

    function x = thresh(t)
        x = zeros(n,1);
        for ii=1:n
            if rho{ii}(1) < t
                x(ii) = 0;
            else
                x(ii) = sum(rho{ii} >= t);
            end
        end
    end

function val = oracle_H(x)
    val = F(x,param.param_F);
end
function val = oracle_R(x)
    x_ctns = interpolate(x, param.param_F.x_lower_vec, param.param_F.x_upper_vec, param.param_F.k_vec);
    val = 0;
    for kk=1:length(x)
        val = val + R_cell{kk}(x_ctns(kk));
    end
end    

vals = zeros(length(lambdas),1);
for ii=1:length(lambdas)
    t = lambdas(ii);
    x = thresh(t);
    vals(ii) = oracle_H(x) + t*(oracle_R(x) - B);
end

end

