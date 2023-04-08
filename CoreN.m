function [core_N,T] = CoreN(X, W, rn)
d=ndims(X);
sze = size(X);
%% Last Core
%W=~isnan(X(:,:,1)); % missing in n-th mode should correspond to W; this is for 3rd order missing of order-3 tensor
Tmat = TT_unfold(X, sze, d-1);
Tmat = Tmat(W(:),:);
[UN,SN,VN] = svds(Tmat,rn);
core_N = reshape(VN', [rn, sze(d), 1]);
%% Compress in n mode
T = NaN(prod(sze(1:end-1)), rn);
T(W(:),:) = UN*SN;
sze(d) = rn;
T = reshape(T, sze);
%% Other cores
end