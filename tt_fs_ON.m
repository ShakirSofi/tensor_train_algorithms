function [G] = tt_fs_ON(Tm, W, ranks)

    d= ndims(Tm);
    sze=size(Tm);
    G = cell(1,d);
    Uon = {};
    % last core and tensor with truncated mode-n tensor
    [core_N, TrM] = CoreN(Tm, W, ranks(d));
    G{d} = core_N;
    
    % other cores
    N=ndims(TrM);
    for n=1:d-1
        sze=size(TrM);
        psz=[n+1:N-1 1:n N];
        Tro = permute(TrM, psz); %[remainaxes, axes of mode-1:n, N]
        nsh = [prod(sze(n+1:N-1)), prod(sze(1:n)), sze(N)];
        Tr= reshape(Tro, nsh);
        x1 = {};
        for k = 1:size(Tr,1)
            Trk = squeeze(Tr(k,:,:));
            x1{k} = Trk;
        end
        Wi=reshape(permute(W, psz(1:N-1)), [nsh(1:end-2) nsh(end-1)]);
        [Uon{n}, Yred]= CR_range(x1, Wi, ranks(n+1));
    end
    %%% finding tt-cores from Uon
    G{1}=reshape(Uon{1}, [1 size(Uon{1})]);
    for ci  = 1:d-2
        Wci = Uon{ci}'*reshape(Uon{ci+1}, prod(sze(1:ci)), sze(ci+1)*ranks(ci+2));
        G{ci+1}=reshape(Wci, ranks(ci+1), sze(ci+1), ranks(ci+2));
    end

end
