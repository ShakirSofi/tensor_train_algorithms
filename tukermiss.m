function [G]= tukermiss(Tmiss, W, r1, r2)
G=cell(1,3);     % T:  I x J x K, core: r1 X J X r2 

[C,~] = CoreN(Tmiss, W, r2); %C: r2 X K

G{3} = C;  
Tmissp = permute(Tmiss, [2 1 3]);
[A]= grid_complete(Tmissp, W.', r1).'; %A : r1 x I 
G{1} = A.';
G{1} = reshape(G{1}, [1, size(G{1})]);
%%{
% mid-core
for j =1:size(Tmiss,2)
    [G{2}(:,j,:)] = kronsol(A.',squeeze(Tmiss(:,j,:)),C);
%}
end