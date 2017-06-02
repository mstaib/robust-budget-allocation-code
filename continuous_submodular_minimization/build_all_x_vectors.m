function x_all = build_all_x_vectors(x_start, is)
    n = length(x_start);
    num_vecs = length(is);
    x_all = zeros(n, num_vecs);
    
    % bug: we should actually allocate n x num_vecs+1 array, with first
    % column equal to xold
    xold = x_start;
    for i=1:num_vecs
        xnew = xold; 
        xnew(is(i)) = xnew(is(i)) + 1;
        
        x_all(:,i) = xnew;
        xold = xnew;
    end
end