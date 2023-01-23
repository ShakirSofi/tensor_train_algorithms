function [Out, IOhm,Ip,YI, SI, P, Ut, Dtot, M, M_c, MM]= PSSM(Ymis, C, W,R2)
%global I J K
[I,J, K] = size(Ymis);

Yred = Ymis;
Yred(isnan(Yred)) = 0;
Yred = tmprod(Yred, C', [3]);

  %% For B
        %% Index Set
        IOhm = {};
        for i=1:I
             % index set 
             Ymis_i = squeeze(Yred(i,:,:));
             if rank(Ymis_i)==R2
                IOhm{i}=i; 
             end
        end     
       % IOhm = IOhm(IOhm~=0);
       IOhm([IOhm{:}]==0)={[]};
       IOhm = IOhm(~cellfun('isempty', IOhm));
        %% SI
             SI= {};
             for i=1:I
                 SI{i}=row_sel(squeeze(Ymis(i,:,:)), 'NaN');
             end
        %%  YI     
             %%{
             YI={};
             for i=1:I
                 YI{i} = squeeze(Yred(i,:,:));
             end
        %% T-SVD
             r2c=cell(1, I);
             r2c(:)={R2};
             % SI*YI
             P = cellfun(@(x,y)x*y, SI,YI, 'UniformOutput',false);
             %Truncated-SVD
             [Ut, ~, ~] = cellfun(@(x) svds(x, R2), P, 'UniformOutput', false);
             
             Ip={};
             for i=1:I
                 Ip{i} = nnz(diag(get_Di(W,i)));
             end  
        %% Dtot and its negative 1/2 power
           Dtot={};
           for i=1:numel(IOhm)
               Dtot{i} = SI{IOhm{i}}'*SI{IOhm{i}};
           end
           %%{
           Dtot = sum(cat(3, Dtot{:}),3);;
           Dtot_1_2 =Dtot^(-0.5);
        %% step 3
            M_c = cellfun(@(x,y)x'*y, SI,Ut, 'UniformOutput',false);
            
            MM = {};
            for i = 1:numel(IOhm)
                MM{i}=M_c{IOhm{i}};
            end
            
            M = Dtot_1_2*cat(2,MM{:});
            
            [ut,~,~] = svds(M, R2);
            tmp = Dtot_1_2*ut;
            [Out, ~] =  qr(tmp,0);
           
           %}
 end