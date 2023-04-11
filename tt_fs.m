function [Cores] = tt_fs(X, W, ranks)
%% Input tensor with missing in last mode
% ranks = [1, r1, r2, 1]
d=ndims(X);
sze = size(X);

assert(d <= 3,'Only for third order tensor')
assert(d+1 == length(ranks),'Ranks are not consistent')

Cores = cell(1,d);
%% Last Core
[Cores{d}, Xmr] = CoreN(X, W, ranks(3));
sze(d)=ranks(d);
%% First Core
n=1; % core number-1
notn = [1:n-1 n+1:d-1];
Xmrp = permute(Xmr, [notn n d]);
Xmrp = reshape(Xmrp, [prod(sze(notn)) sze(n) sze(d)]);
Wn = permute(W, [notn n]);
Wn = reshape(Wn, [prod(sze(notn)) sze(n)]);
[Cores{n}] = grid_complete(Xmrp, Wn, ranks(n+1));

%% Other cores, solving frame eqn with least square
% Vec(X) = Frame_Matrix * vec(Core_2); 
% where Frame_Matrix = kron(Core1,eye(size(X,2)),Core_3)

Cores{1} = reshape(Cores{1}, 1,sze(1),ranks(2));

% inital guess or core 2
G20 = tmprod(rep0(Xmr), squeeze(Cores{1}).', 1);
% left and right interfacte matricies
G_nl =squeeze(Cores{1}); G_nr = Cores{3};  

FME = kron(G_nr', eye(sze(2))); 
FME = kron(FME, G_nl);

%% Solve linear system with some missing values in b, A*x = b
tol=1e-4;
maxit=5;
x0 = reshape(G20, [],1);
b_known = TT_unfold(X,size(X), d);  % missing values as nan in this vector
%tic
[core_mid_opt,~, ~,~] = solve_LS_miss(FME, b_known, x0, tol, maxit);
%toc

Cores{2}= reshape(core_mid_opt, ranks(2), sze(2), ranks(3));
end