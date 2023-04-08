function [Out_c, Yred]= CR_range(Ymis, W, R)
       I = length(Ymis);
        IOhm = {};
        for i=1:I
             % index set 
             if nnz(diag(get_Di(W,i)))> R
                IOhm{i}=i; 
             end
        end     
       IOhm([IOhm{:}]==0)={[]};
       IOhm = IOhm(~cellfun('isempty', IOhm));

             % SI
             SI= {};
             for i=1:I
                 SI{i}=row_sel(Ymis{i}, 'NaN');
             end

             YI={};
             for i=1:I
                 YI{i} = rep0(Ymis{i});
             end
             
             
            % U_|_
             P = cellfun(@(x,y)x*y, SI,YI, 'UniformOutput',false);
            %%{
             [Ut, ~, ~] = cellfun(@(x) svd(x), P, 'UniformOutput', false);
             Ip={};
             for i=1:I
                 Ip{i} = nnz(diag(get_Di(W,i)));
             end    
             U = cellfun(@(x, Ip) x(:,1:Ip), Ut,Ip, 'UniformOutput',false);

             r2c=cell(1, I);
             r2c(:)={R};
             U_c = cellfun(@(x,r2c) x(:,r2c+1:end), U, r2c, 'UniformOutput',false);
             %%{
            M_c = cellfun(@(x,y)x'*y, SI,U_c, 'UniformOutput',false);
            
            MM = {};
            for i = 1:numel(IOhm)
                MM{i}=M_c{IOhm{i}};
            end
            
            % Left Null space concatenate basis
            M = cat(2,MM{:});
            [ut,~,~] = svd(M);
            Out_c = ut(:,end-R+1:end);

            % Updated core
            Yred = cellfun(@(x) Out_c.'*rep0(x), Ymis, 'UniformOutput', false);
end