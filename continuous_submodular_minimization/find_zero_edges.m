function [inx_left, inx_right] = find_zero_edges(func, vals, thresh)
    inx_zero = ternary_search(func, vals);
    
    % because sometimes our array is basically monotone and it gets tripped
    % up by noise
    if func(vals(end)) == 0
        inx_zero = length(vals);
    elseif func(vals(1)) == 0
        inx_zero = 1;
    end
    
    % get down to value within thresh of zero, then do linear search to
    % take in account the fact that the function is not perfectly monotone
    % near zero. Smaller threshold values will do less linear searching and
    % therefore be much faster
    if nargin < 3
        thresh = 1e-5; % default value
    end
    
    inx_left_left = 1;
    inx_left_right = inx_zero;
    while inx_left_right > inx_left_left + 1
        inx_curr = floor((inx_left_left+inx_left_right)/2);
        f_curr = func(vals(inx_curr));
        if f_curr <= thresh
            inx_left_right = inx_curr;
        else
            inx_left_left = inx_curr;
        end
    end
    
    inx_left = inx_left_right;
    while func(vals(inx_left)) > 0
        inx_left = inx_left + 1;
    end
    
    inx_right_left = inx_zero;
    inx_right_right = length(vals);
    while inx_right_right > inx_right_left + 1
        inx_curr = floor((inx_right_left+inx_right_right)/2);
        f_curr = func(vals(inx_curr));
        if f_curr > thresh
            inx_right_right = inx_curr;
        else
            inx_right_left = inx_curr;
        end
    end
    
    inx_right = inx_right_left;
    while func(vals(inx_right)) > 0
        inx_right = inx_right - 1;
    end
end

function [inx] = ternary_search(func, vals)
% returns an index within the array vals for which the function attains
% its minimum (in this case should be zero)
%
% adapted from http://chaoxuprime.com/posts/2013-07-27-find-the-minimum-of-an-array.html
%
% we will need to be more sophisticated: we care about indices where <= 0,
% not just the strict minimum...

left = 1;
right = length(vals);

while true
    ml = floor((2*left+right)/3);
    m = floor((left+right)/2);
    mr = floor((left+2*right)/3);
   
    f_ml = func(vals(ml));
    f_m = func(vals(m));
    f_mr = func(vals(mr));
      
   if right-left < 6 || (f_ml == f_m && f_m == f_mr)
       [Y,I] = min(arrayfun(func, vals(left:right))); %earlier they had right+1 here??
       inx = I + left - 1;
       return;
   end
   
   if f_ml >= f_mr
       left = ml;
   end
   
   if f_ml <= f_mr
       right = mr;
   end
end

end