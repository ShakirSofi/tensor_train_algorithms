function [A, B, G]= tuker2_miss(Tmiss, W, r1, r2)

[B] = grid_complete(Tmiss, W, r2).'; % r2 X J
Tmissp = permute(Tmiss, [2 1 3]);
[A]= grid_complete(Tmissp, W.', r1).'; %A = r1 x I 

% core
S3 = row_sel(tens2mat(Tmiss,3).');
Ysub3 = tens2mat(Tmiss,3);
Ysub3 = Ysub3(:, W(:));

H = kron(B,A)*S3.';
G3 = Ysub3*pinv(H);
G = reshape(G3.', r1, r2,[]);
% Return 
A=A.'; B=B.';
end