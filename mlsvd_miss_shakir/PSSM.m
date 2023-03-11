function [Out, IOhm,Ip,YI, SI, P, Ut, Dtot, M, M_c, MM]= PSSM(Ymis, Yred, W,R)
%global I J K
[I,~, ~] = size(Ymis);

  %% For A
        %% Index Set
        IOhm = {};
        for i=1:I
             % index set 
             if nnz(diag(get_Di(W,i)))> R
                IOhm{i}=i; 
             end
        end     
       IOhm([IOhm{:}]==0)={[]};
       IOhm = IOhm(~cellfun('isempty', IOhm));
        %% SI
             SI= {};
             for i=1:I
                 SI{i}=row_sel2(squeeze(Ymis(i,:,:)), 'NaN');
             end
        %%  YI     
             %%{
             YI={};
             for i=1:I
                 YI{i} = squeeze(Yred(i,:,:));
             end
        %% T-SVD
             r2c=cell(1, I);
             r2c(:)={R};
             % SI*YI
             P = cellfun(@(x,y)x*y, SI,YI, 'UniformOutput',false);
             %Truncated-SVD
             [Ut, ~, ~] = cellfun(@(x) svds(x, R), P, 'UniformOutput', false);
             
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
           Dtot = sum(cat(3, Dtot{:}),3);
           Dtot_1_2 =Dtot^(-0.5);
        %% step 3
            M_c = cellfun(@(x,y)x'*y, SI,Ut, 'UniformOutput',false);
            
            MM = {};
            for i = 1:numel(IOhm)
                MM{i}=M_c{IOhm{i}};
            end
            
            M = Dtot_1_2*cat(2,MM{:});
            
            [ut,~,~] = svds(M, R);
            tmp = Dtot_1_2*ut;
            [Out, ~] =  qr(tmp,0);
           
           %}
 end