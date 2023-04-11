function [x_opt, Flag, A, b] = solve_LS_miss(A, b_known, x0, tol, maxit)

warning('off','all')

sze= size(b_known,1);
% Solving linears systems of equation with missing values in b
% Ax=b
missing_indices = isnan(b_known);
% Solve the system with the known values of b
A_known = A(~missing_indices, :);
b_known = b_known(~missing_indices);
%x_known = A_known \ b_known;
[x_known,~,~] = bicg(A_known'*A_known, A_known'*b_known, tol, maxit, [], [], x0);
% Substitute the solutions into the system with the missing values of b
b_missing = zeros(size(sze));
b_missing(missing_indices) = A(missing_indices, :) * x_known;
% Combine the known and missing values of b
b = zeros(sze,1);
b(~missing_indices) = b_known;
b(missing_indices) = b_missing(missing_indices);
% Solve the system with the complete values of b
%[x_opt, Flag, res] = lsqr(A, b,tol, maxit, [], [], x0);
[x_opt, Flag, res] = bicg(A'*A, A'*b, tol, maxit, [], [], x0);
disp(res)
end