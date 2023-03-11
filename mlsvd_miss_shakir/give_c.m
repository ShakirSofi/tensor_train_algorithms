function [C, Srow3, Yred, Ysub] = give_c(Ymis, R3)

%% parameters
if nargin < 2 || isempty(R3)
    R3 = 2;
end
%%
Y3 = tens2mat(Ymis,3);
Srow3 =row_sel(Y3', 'NaN');
Ysub = Srow3*rep0(Y3)';
[C, ~, ~ ] =svd(Ysub', 'econ');
C= C(:,1:R3);
Yred = mat2tens(C'*rep0(Y3), [size(Ymis, 1), size(Ymis, 2), R3],[3]);
end
