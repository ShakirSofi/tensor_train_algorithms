function [G] = tt_fs_gc(Tm, W, ranks)

d= ndims(Tm);
sze=size(Tm);
G = cell(1,d);

% last core and tensor with truncated mode-n tensor
[core_N, TrM] = CoreN(Tm, W, ranks(d-1));

% other cores
N=ndims(TrM);
for n=1:d-2
    sze=size(TrM);
    psz=[n+1:N-1 1:n N];
    Tro = permute(TrM, psz); %[remainaxes, axes of mode-1:n, N]
    nsh = [prod(sze(n+1:N-1)), prod(sze(1:n)), sze(N)];
    Tr= reshape(Tro, nsh);
    Wi=reshape(permute(W, psz(1:N-1)), [nsh(1:end-2) nsh(end-1)]);
    [Out_c]= grid_complete(Tr, Wi, ranks(n+1));
    if n==d-2
        Yred = tmprod(rep0(Tr), Out_c.', 2);
    end
    or = reshape(Out_c, [sze(1:n), ranks(n+1)]); 
    if n==1
       orr = reshape(or, [1 size(or)]);
    else
       Gdot = givedot(G,n-1);  
       orr = tensorprod(squeeze(Gdot), or, [1:n-1], [1:n-1]);
    end 
    G{n}= orr;
end
   
    G{d-1} = permute(Yred, [2 1 3]);
    % last core
    G{d} = core_N;
    % second last core/ by solving ls problem on cores  
    G{d-1} = update_gn_core(G, Tm, d-1);
end