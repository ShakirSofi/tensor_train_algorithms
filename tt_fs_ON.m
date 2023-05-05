function [G] = tt_fs_ON(Tm, W, ranks)
    d= ndims(Tm);
    sze=size(Tm);
    clear G Uon
    G = cell(1,d);
    Uon = {};
    % last core and tensor with truncated mode-n tensor
    [core_N, TrM] = CoreN(Tm, W, ranks(d));
    % other cores
    N=ndims(TrM);
    for n=1:d-1
        sze=size(TrM);
        psz=[n+1:N-1 1:n N];
        Tro = permute(TrM, psz); %[remainaxes, axes of mode-1:n, N]
        nsh = [prod(sze(n+1:N-1)), prod(sze(1:n)), sze(N)];
        Tr= reshape(Tro, nsh);
        Wi=reshape(permute(W, psz(1:N-1)), [nsh(1:end-2) nsh(end-1)]);
        [Uon{n}]= grid_complete(Tr, Wi, ranks(n+1));
        if n==d-2
           Yred = tmprod(rep0(Tr), Uon{n}.', 2);
        end
    end
    %%% finding tt-cores from Uon
    G{1}=reshape(Uon{1}, [1 size(Uon{1})]);
    for ci  = 1:d-2
        Wci = Uon{ci}'*reshape(Uon{ci+1}, prod(sze(1:ci)), sze(ci+1)*ranks(ci+2));
        G{ci+1}=reshape(Wci, ranks(ci+1), sze(ci+1), ranks(ci+2));
    end
    G{d-1} = permute(Yred, [2 1 3]);
    % last core
    G{d} = core_N;
    % second last core/next to last
    G{d-1} = update_gn_core(G, Tm, d-1);
end

