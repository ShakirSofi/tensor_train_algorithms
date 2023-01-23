function [A, B, G] = P_V_D(X, R1, R2)

X_cell={};
for i =1:size(X,3)
    X_cell{i} = X(:,:,i);    
end
[Uk, Sk, Vk] = cellfun(@(x) svd(x), X_cell, 'UniformOutput', false); 

U = cellfun(@(x,y)x*y, Uk,Sk, 'UniformOutput',false);
V = cellfun(@(x,y)x*y', Vk,Sk, 'UniformOutput',false);

U = cat(2,U{:});
V = cat(2,V{:});

[A, ~, ~] = svds(U, R1);
[B, ~, ~] = svds(V, R2);

G = cellfun(@(x)A'*x*B, X_cell, 'UniformOutput',false);
G= cat(3, G{:});
end