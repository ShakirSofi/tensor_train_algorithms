function[Cores,T,  Tmiss, W, ranks, UN]= tt_ts_test
R1=2;R2=2;R3=4;
% generate
[Uo,So] = lmlra_rnd([10 12 20], [R1 R2 R3]);
T = tmprod(So,Uo,[1 2 3]);
% remove some fibres, say 20% of tube fibres
[Tmiss,~,~] = give_miss_3(T, 0.35);
X=Tmiss;
ranks = [2 3];
d=ndims(X);
sze = size(X);
Cores = cell(1,d);
%% Last Core
W=~isnan(X(:,:,1));
Tmat = TT_unfold(X, sze, d-1);
Tmat = Tmat(W(:),:);
[UN,SN,VN] = svds(Tmat,ranks(end));
Cores{d} = reshape(SN*VN', [ranks(end), sze(d), 1]);
end