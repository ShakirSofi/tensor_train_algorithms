function [X] = kronsol(A,G,B)   % solve for X: G = A.X.B
% G has missing rows, A and B are fully observed
% Find the indices of the available rows
available_rows = find(~isnan(sum(G, 2)));
% Construct the modified equation
Gp = G(available_rows, :);
Ap = A(available_rows, :);
Bp = B;
%X = pinv(Ap)*Gp*pinv(Bp);
Hinv=kron(pinv(Bp.'), pinv(Ap));
Gv=reshape(Gp, [],1);
X_vec = Hinv*Gv;
X = reshape(X_vec, size(A,2), size(B,1));
end