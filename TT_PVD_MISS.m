function [Ucores] = TT_PVD_MISS(X, W, ranks)
 d = ndims(X); %order of tensor
 sze = size(X); 
 % ranks =[1 r1 r2 r3.. 1]
 %% initilizing
 Ucores = cell(1, d); 
 r = ranks(2:d);

 nr= 1;
 pr= length(r);
 
 n = 2;
 p=d-1;
 
 %% Compute First and Last core;
W3 = reshape(W, sze(1),[]);
Tm3 = reshape(X , sze(1), [], sze(d));
cores3 = tukermiss(Tm3, W3, r(nr), r(pr));% with lq_tuk2
%cores3 = tt_fs_3(Tm3, W3, [1 r(nr) r(pr) 1]); % very expensive ...
Ucores{1} = cores3{1};
Ucores{end} = cores3{3};
G=cores3{2};
[G,~] = prmt(G);
%disp(size(G))
 %% Compute other cores from G
 %%{
 if (d > 3)
         while (n~=p)  
             [G,~] = prmt(G);
              if ismatrix(squeeze(reshape(G, r(nr)*sze(n), [], r(pr)*sze(p))))   
                 G = reshape(G, r(nr)*sze(n), sze(p), r(pr));
                  nr = nr+1;
                  pr = pr-1;
                 [G,~] = prmt(G);
                 [Un,Up, G] = P_V_D(G, r(nr), r(pr));
                 Ucores{n} = reshape(Un, r(nr-1), sze(n), r(nr));
                 [G,~] = prmt(G);
                 Ucores{p} = tmprod(G,Up,[3]); %G x_3 Up;
                 break;
              else
                 G = reshape(G, r(nr)*sze(n), [], r(pr)*sze(p));
              end
             nr = nr+1;
             pr = pr-1;
             [G,~] = prmt(G);
             [Un,Up, G] = P_V_D(G, r(nr), r(pr));

             Ucores{n} = reshape(Un, r(nr-1), sze(n), r(nr));
             %disp('Un reshaped')
             %size(Ucores{nr})
             
             Ucores{p} = reshape(Up', r(pr), sze(p), r(pr+1));
             %disp('Up reshaped')
             %size(Ucores{pr})
             n=n+1;
             p=p-1;
         end
 end     
  %}
  if mod(d,2)~=0
     [Ucores{n},~] = prmt(G);
  end
  
    function [y, sp] = prmt(yi)
        sN = size(yi);
        N = ndims(yi);
        szt = sN(2:end-1);
        sN(2:end-1) = [];
        sp =[sN szt];
        y = permute(yi, [1 N 2:N-1]);
    end
end