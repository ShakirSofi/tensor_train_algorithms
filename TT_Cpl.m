function [G] = TT_Cpl(Tmiss, G0, ttranks, tol, maxit)
% tt_ranks = [1 r1, r2, ...1]
ranks = ttranks(2:end-1);
sze=size(Tmiss);
d = length(sze);
% initilization cores;
G=G0;
v_T = TT_unfold(Tmiss,size(Tmiss),3);
for i = 1:2
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
    %vh = pinv(FME)*v_T;
    x0 = reshape(G{n}, [],1);
    vh = solve_LS_miss(FME, v_T, x0, tol, maxit);
    if n==1  % For G{1}
       G{n} = reshape(vh,1, sze(n), ranks(n));
    elseif n==d  % For G{d}
       G{n} = reshape(vh,ranks(n-1), sze(n), 1);
    else  %other
       G{n} = reshape(vh,ranks(n-1), sze(n), ranks(n));
    end
end
end
end