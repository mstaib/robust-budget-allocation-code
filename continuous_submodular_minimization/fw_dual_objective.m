function val = fw_dual_objective(w, step, direction, weights)
    %wnew = cellfun(@(x,y) (1-step)*x + step*y, w, direction, 'UniformOutput', false);
    n = length(w);
    wnew = cell(n,1);
    
    for ii=1:n
        wnew{ii} = w{ii} + step*direction{ii};
    end
    rho = grad_w(wnew, weights);
    
    val = cell_innerprod(wnew, rho) + regularize_sum(rho, weights); %sum(cellfun(@(x,z) 0.5*z*sum(x.^2), rho, weights));
end

function val = regularize_sum(rho, weights)
    val = 0;
    for kk=1:length(weights)
        val = val + 0.5*sum(rho{kk}.^2 .* weights{kk});
    end
end

function val = cell_innerprod(w, rho)
    val = 0;
    for ii=1:length(w)
        val = val + w{ii}(:)'*rho{ii}(:);
    end
    %val = sum(cellfun(@(wi,rhoi) wi(:)'*rhoi(:), w, rho));
end

function rho_i = grad_w_block(w_i, weights_i)
    rho_i = -pav(weights_i.^(-1) .* w_i, weights_i);
end

function rho = grad_w(w, weights)
    n = length(w);
    rho = cell(n,1);
    for i=1:n
        rho{i} = grad_w_block(w{i}, weights{i});
    end
end