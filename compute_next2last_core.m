function Gm = compute_next2last_core(Tmred, GL)
% GL: ... x rn-1; GR: rn x ...   ; Tmiss:  I_1 x I_2 x r_n2
r_n1 = size(GL,2);  I_n = size(Tmred,2);  r_n2 = size(Tmred,3); 
Gm = zeros(r_n1, I_n, r_n2);

for ii=1:I_n
    slice = squeeze(Tmred(:,ii,:));
    xout = Solve_ortho(GL, slice);
    Gm(:,ii,:) = xout(1:size(GL,2),:); 
end

function [xout] = Solve_ortho(A, b_known)
[r,c] = size(b_known);
szA = size(A);
idx = isnan(b_known(:,1));
b_known(isnan(b_known)) = 0;
Uex =diag(idx);
Uex(:,all(Uex==0))=[];
Aex = [A Uex]; % extend standard basis
Q = ortho_full(Aex, szA(2)+1); % orthonormalize
R = Q'*Aex;  % find R.

if c >= 2
    yout = Q.'*rep0(b_known);
else
    yout = zeros(size(Q,2), c);

    for ic = 1:c
        yout(:,ic) = Q.'*rep0(b_known);
    end
end

xout  = pinv(R+1e-16*eye(size(R)))*yout;

end

% orthonormalize fully.
function orthonormalized_matrix = ortho_full(matrix, k)
    % Input:
    % matrix: Partial orthonormalized, upto k.
    % Output:
    % matrix: Fully orthonormalized.
    [m, n] = size(matrix);
    orthonormalized_matrix = matrix; % Initialize the result matrix.
    for i = k:n  
        v = matrix(:, i); 
        % Orthogonalize with respect to previous columns.
        for j = 1:i-1
            u = orthonormalized_matrix(:, j); 
            v = v - (u' * v) / (u' * u) * u; 
        end
        orthonormalized_matrix(:, i) = v / norm(v);
    end
end

end 