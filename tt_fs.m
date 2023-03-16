function [Cores, UN,  T, sze] = tt_fs(X, ranks)
d=ndims(X);
sze = size(X);
Cores = cell(1,d);
%% Last Core
W=~isnan(X(:,:,1));
Tmat = TT_unfold(X, sze, d-1);
Tmat = Tmat(W(:),:);
[UN,SN,VN] = svds(Tmat,ranks(end));
Cores{d} = reshape(SN*VN', [ranks(end), sze(d), 1]);

%% Compress in last mode
%{
Tmat = Tmat*squeeze(Cores{d})';
T = zeros(prod(sze(1:end-1)), ranks(end));
T(W(:),:) = Tmat;
sze(d) = ranks(end);
size(T)
T = TT_fold(T, sze, d);

% T =  tmprod(rep0(X), Cores{d}, d); ---alter
%sze(d) = ranks(end);
%}

%% Other cores

end