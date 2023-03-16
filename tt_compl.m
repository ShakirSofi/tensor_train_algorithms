function[G] = tt_compl
%{
sze = [3 4 5 6];
T=reshape(1:prod(sze), sze);
ranks = [2 2 3];
[Ucore]= TT_S_V_D(T, ranks);
n=2;k=1;

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

% Simple Example
sze = [3 4 5 6];
ranks = [2 2 3];
d= length(sze);
T=reshape(1:prod(sze), sze);
% initilization cores;
G0=initcoreten(sze,ranks);
G=G0;
%n=2;
%k=2;
v_T = reshape(T, [],1);

for n=1:d
    % Mode splitting
    [G_nl,G_nr]=G_nmode_devide(G,n);
   
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
    vh = pinv(FME)*v_T;

    if n==1  % For G{1}
       G{n} = reshape(vh,1, sze(n), ranks(n));
    elseif n==d  % For G{d}
       G{n} = reshape(vh,ranks(n-1), sze(n), 1);
    else  %other
       G{n} = reshape(vh,ranks(n-1), sze(n), ranks(n));
    end
    
end

end