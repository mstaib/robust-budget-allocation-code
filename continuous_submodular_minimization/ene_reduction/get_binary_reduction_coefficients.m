function [ a ] = get_binary_reduction_coefficients( n, eps )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

a = get_binary_reduction_coefficients_no_eps(n);

% if a target epsilon was supplied, cut down on the values a_i until each
% of them is less than eps*n
if nargin >= 2
    % edge case: if caller asks for all a_i <= eps*n < 1/n*n = 1, just
    % return n ones
    if eps < 1/n
    	a = ones(1,n);
        return;
    end
    
    a = sort(a);
    while a(end) > eps*n
        a = [a(1:end-1) a(end)/2 a(end)/2];
        a = sort(a);
    end
end
end

function [ a ] = get_binary_reduction_coefficients_no_eps( n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

binrep = dec2bin(n);
m = length(binrep)-1;

a(1) = 1;
for ii=2:m+1
    a(ii) = 2^(ii-2);
end

binrep_nums = fliplr(arrayfun(@str2num, binrep));
powers = arrayfun(@(ii) 2^ii*binrep_nums(ii+1), 0:m);

powers = powers(1:end-1); %prune biggest power or sum will be too big

a = [a powers(powers > 0)];


end

