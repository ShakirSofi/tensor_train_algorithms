function [X] = TT_unfold( X, dim, i )
X = reshape(X, prod(dim(1:i)), []);
end