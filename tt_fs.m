function [Cores] = tt_fs(X, W, ranks)
%% Input tensor with missing in last mode
% ranks = [1, r1, r2, r3, 1]
d=ndims(X);
sze = size(X);

assert(d <= 3,'Only for third order tensor')
assert(d+1 == length(ranks),'Ranks are not consistent')

Cores = cell(1,3);
%% Last Core
[Cores{3}, Xmr] = CoreN(X, W, ranks(3));
%% First Core
Xmis1 = {};
for i =1:size(Xmr,2)
    Xmis1{i}=squeeze(Xmr(:,i,:));
end
[Cores{1}, ~]=CR_range(Xmis1, W', ranks(2));
%% Other cores, solving frame eqn with least square
% Vec(X) = Frame_Matrix * vec(Core_2); 
% where Frame_Matrix = kron(Core1,eye(size(X,2)),Core_3)
Cores{1} = reshape(Cores{1}, 1,sze(1),ranks(2));
G_nl =squeeze(Cores{1}); G_nr = Cores{3};   

FME = kron(G_nr', eye(sze(2))); 
FME = kron(FME, G_nl);
%% Solve linear system with some missing values in b, A*x = b
tol=1e-4;
maxit=10;
%
b_known = TT_unfold(X,size(X), d);  % missing values as nan in this vector
[core_mid_opt,~, ~,~] = solve_LS_miss(FME, b_known, tol, maxit);
Cores{2}= reshape(core_mid_opt, ranks(2), sze(2), ranks(3));
end