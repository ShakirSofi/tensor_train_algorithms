function [Ucores] = TT_PVD(X, ranks)
 d = ndims(X); %order of tensor
 sze = size(X);
 %% initilizing
 Ucores = cell(1, d); 
 r = ranks;

 nr= 1;
 pr= length(ranks);
 
 n = 2;
 p=d-1;
 
 %% Compute First and Last core;
  [X,sp] = prmt(X);
  Xr = reshape(X, sp(1), sp(2), prod(sp(3:end)));
  [U1,Uend, G] = P_V_D(Xr, r(nr), r(pr));
  Ucores{1}=reshape(U1, 1,size(U1,1), size(U1,2));
  Ucores{end} = Uend';
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