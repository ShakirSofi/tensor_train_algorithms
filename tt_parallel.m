function [G] = tt_parallel(T,ranks)
    d= ndims(T);
    sze=size(T);
    G = cell(1,d);
    Uon = {};
    %%% On cols
    for n=1:d-1
        Xn = TT_unfold(T, sze, n); 
        [Uon{n}, s, v]= svds(Xn, ranks(n+1));
    end
    %%% finding tt-cores from Uon
    G{1}=reshape(Uon{1}, [1 size(Uon{1})]);
    for ci  = 1:d-2
        Wci = Uon{ci}'*reshape(Uon{ci+1}, prod(sze(1:ci)), sze(ci+1)*ranks(ci+2));
        disp(size(Wci))
        G{ci+1}=reshape(Wci, ranks(ci+1), sze(ci+1), ranks(ci+2));
        disp(size(G{ci+1}))
    end
    G{d} = s*v.';
end
