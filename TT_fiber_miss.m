function [G] = TT_fiber_miss(Tm, W, ranks)
%%% Main funx
[Uon] = compute_ranges(Tm, W, ranks);
[G] = compute_cores(Tm, Uon, ranks);

%%%% Utility Functions
function [Uon] = compute_ranges(Tm, W, ranks)
    clear Uon
    Uon = {};
    N=ndims(Tm);
    for n=1:N-1
        sze=size(Tm);
        psz=[n+1:N-1 1:n N];
        Tro = permute(Tm, psz); %[remainaxes, axes of mode-1:n, N]
        nsh = [prod(sze(n+1:N-1)), prod(sze(1:n)), sze(N)];
        Tr= reshape(Tro, nsh);
        Wi=reshape(permute(W, psz(1:N-1)), [nsh(1:end-2) nsh(end-1)]);
        Uon{n}= grid_complete_principal(Tr, Wi, ranks(n+1));
    end
end

function [G] = compute_cores(Tm, Uon, ranks)
    N= ndims(Tm);
    sze=size(Tm);
    G = cell(1,N);
    %%% finding tt-cores from Uon
    G{1}=reshape(Uon{1}, [1 size(Uon{1})]);
    for ci  = 1:N-2
        Wci = Uon{ci}'*reshape(Uon{ci+1}, prod(sze(1:ci)), sze(ci+1)*ranks(ci+2));
        G{ci+1}=reshape(Wci, ranks(ci+1), sze(ci+1), ranks(ci+2));
    end
    % last core
    Y = tens2mat(Tm, N);
    % find columns that do not contain NaN values
    cols = all(~isnan(Y));
    % select those columns
    Ysub = Y(:, cols).';
    [us, ss, vs] = svds(rep0(Y.'), ranks(end-1));
    G{N} = vs.';

    % Update G{N-1}
    % 3rd-order reshaping
    T3r = reshape(Tm, prod(sze(1:N-2)), sze(N-1), []);
    % Mode splitting
    [G_nl,G_nr] = G_nmode_devide(G, N-1);
    G_nl = reshape(G_nl, [], ranks(N-1));
    G_nr = reshape(G_nr, ranks(N), []); 
    Tmred = tmprod(T3r, G_nr, 3);
    G{N-1} = give_mid_cr(Tmred, G_nl);
    %}
end
end