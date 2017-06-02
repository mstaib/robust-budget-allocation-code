function [ G, elem_weights, t_vec, mapping ] = map_lattice_to_set( F, weights, k_vec, eps)
%MAP_LATTICE_TO_SET Summary of this function goes here
%   Detailed explanation goes here

% F is a lattice-submodular function defined on the product of
% {0,...,k_i-1} for k_i in k_vec

if nargin < 4
    eps = 1;
end

n = length(k_vec);
a_cell = cell(1,n);
for ii=1:n
    a_cell{ii} = get_binary_reduction_coefficients(k_vec(ii) - 1, eps);
end

t_vec = cellfun(@(a_i) length(a_i), a_cell);
A_mat = build_mapping_mat_from_cell(a_cell);
mapping = @(x) A_mat * x;
%mapping = @(x) mapping_gen(x, a_cell);
G = @(x,param_F) F(mapping(x),param_F);

elem_weights_cell = cellfun(@(x,y) x*y, num2cell(weights(:)), a_cell(:), 'UniformOutput', false);
elem_weights = [elem_weights_cell{:}];
end

function [ G, t_vec, mapping ] = map_lattice_to_set_no_weights( F, k_vec )
%MAP_LATTICE_TO_SET Summary of this function goes here
%   Detailed explanation goes here

% F is a lattice-submodular function defined on the product of
% {0,...,k_i-1} for k_i in k_vec

n = length(k_vec);
a_cell = cell(1,n);
for ii=1:n
    a_cell{ii} = get_binary_reduction_coefficients(k_vec(ii) - 1);
end

t_vec = cellfun(@(a_i) length(a_i), a_cell);
A_mat = build_mapping_mat_from_cell(a_cell);
mapping = @(x) A_mat * x;
%mapping = @(x) mapping_gen(x, a_cell);
G = @(x) F(mapping(x));

end

function A_mat = build_mapping_mat_from_cell(a_cell)
    m = length(a_cell);
    n = sum(cellfun(@length, a_cell));
    
    A_mat = spalloc(m, n, 0);
    
    start_inx = 1;
    for ii=1:m
        t = length(a_cell{ii});
        elem_range = start_inx:start_inx + t-1;
        
        A_mat(ii,elem_range) = a_cell{ii};
        start_inx = start_inx + t;
    end
end