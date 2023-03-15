function[Cores] = tt_Frame_Eq(T, ranks)
% TT-Decomposition using frame equation.

%{
% Mode splitting
[G_nl,G_nr]=G_nmode_devide(Ucore,n);
GL_m = reshape(G_nl, [], ranks(n-1));
GR_m = reshape(G_nr, ranks(n), []);
Xk = GL_m*squeeze(Ucore{n}(:,k,:))*GR_m;
Xraw_k=squeeze(T(:,k,:));
% Ensure X_raw_k=Xk;

v_T = reshape(T, [],1);
v_Xn = reshape(Ucore{n}, [], 1);

FME = kron(GR_m', eye(sze(n))); %I_n: size(sze(n)); Frame matrix 
FME = kron(FME, GL_m);
vh = FME*v_Xn;
%}

% initilization cores;
sze=size(T);
d=numel(sze);
G0=initcoreten(sze,ranks);
Cores=G0;
%n=2;
%k=2;
v_T = reshape(T, [], 1); % Vectorization of input tensor
maxiter=3;
for iter=1:maxiter
for n=1:d
    % Mode splitting
    [G_nl,G_nr]=G_nmode_devide(Cores,n);
   
    if n==1  % For G{1}
        GL_m = G_nl; %reshape(G_nl, [], 1);
        GR_m = reshape(G_nr, ranks(1), []);

    elseif n==d  % For G{d}
        GL_m = reshape(G_nl, [], ranks(n-1));
        GR_m = G_nr; %reshape(G_nr, ranks(d), []);

    else        %other
        GL_m = reshape(G_nl, [], ranks(n-1));
        GR_m = reshape(G_nr, ranks(n), []);
    end
    %v_Gn = reshape(G{n}, [], 1);
    FME = kron(GR_m', eye(sze(n))); %I_n: size(sze(n)); Frame matrix 
    FME = kron(FME, GL_m);
    %vh = pinv(FME)*v_T;
    vh = lsqr(FME,v_T);
    %vh = cgs(FME.'*FME, FME.'*v_T); % cgs on normal eqns
    if n==1  % For G{1}
       Cores{n} = reshape(vh,1, sze(n), ranks(n));
    elseif n==d  % For G{d}
       Cores{n} = reshape(vh,ranks(n-1), sze(n), 1);
    else  %other
       Cores{n} = reshape(vh,ranks(n-1), sze(n), ranks(n));
    end
    
end
end
end