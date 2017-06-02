function [is, js] = build_increment_ordering(rho)
    n = length(rho);
    k_vec = cellfun(@length,rho);

    % first order all rhos (does preserve the ordering within rows if equal
    % values)
    all_rhos = vertcat(rho{:}); % we are now assuming rho is made up of colvecs
    num_total_rhos = length(all_rhos);
    [~, s] = sort(all_rhos, 'descend');

    is_orig = zeros(1,num_total_rhos); js_orig = zeros(1,num_total_rhos);
    k_cum = 0;
    for i=1:n
        is_orig(1 + k_cum : k_vec(i) + k_cum) = i;
        js_orig(1 + k_cum : k_vec(i) + k_cum) = 1:k_vec(i);
        k_cum = k_cum + k_vec(i);
    end
    is = is_orig(s);
    js = js_orig(s);
end