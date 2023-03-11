function[G3] = get_core(A, B, C, Srow3,Ysub, R1,R2,R3)
% solve for G: MG=N
M = Srow3*kron(B,A);
N = Ysub*C;

size(M)
size(N)
G3 = M \ N; 
G3 = reshape(G3, [R1,R2,R3]);
size(G3)
end