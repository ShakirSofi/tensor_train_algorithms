function [Out, IOhm]= STMC(Ymis, C, W,index, R1,R2,R3)
%global I J K
[I,J, K] = size(Ymis);
 

Ymis(isnan(Ymis)) = 0;
Yred = tmprod(Ymis, C', [3]);

Ymis =Yred;
  %% For B
if index==2
        IOhm = [];

        for i=1:I
             % index set 
             Ymis_i = Ymis(i,:,:);
             Ymis_i(isnan(Ymis_i)) = 0;
             %Ymis_i = reshape(Ymis_i, J, K);
             Ymis_i = squeeze(Ymis_i);
             %Di = get_Di(W,i);
             if rank(Ymis_i)==R2
                IOhm(i)=i; 
             end
        end     
        IOhm = IOhm(IOhm~=0); 
             % SIohm
             SIOhm= {};

             for i=1:numel(IOhm)
                 %SIOhm{i}=row_sel(reshape(Ymis(IOhm(i),:,:), J, K));
                 SIOhm{i}=row_sel(squeeze(Ymis(IOhm(i),:,:)), 'NaN');
             end

             % Yred
             %%{
             YredOhm ={};
             for i=1:numel(IOhm)
                 %Di = get_Di(W,IOhm(i));
                 Ymis_i = Ymis(IOhm(i),:,:);
                 Ymis_i(isnan(Ymis_i)) = 0;
                 %disp(size(Ymis_i))
                 %disp(size(C))
                 YredOhm{i}= squeeze(Ymis_i);
             end
             %out = cat(2,Yred{:});
             %}

            %YredOhm = squeeze(num2cell(Yred,[2 3]));
            % U_|_
             P = cellfun(@(x,y)x*y, SIOhm,YredOhm, 'UniformOutput',false);
             
            Us = {};
            for i =1:numel(P)
                Ip = nnz(diag(get_Di(W,IOhm(i))));
                %size(P{i})
                [up, ~,~] = svd(P{i});
                Us{i}=up(:,Ip-R2+1:Ip);
                size(Us{i})
            end
             
            %[U, ~, ~] = cellfun(@(x) svd(x), P, 'UniformOutput', false); 
            %U_c = cellfun(@(x) x(:, Ii-R2+1:end), U, 'UniformOutput', false);

            M_c = {};
            for i =1:numel(Us)
                M_c{i} = SIOhm{i}'*Us{IOhm(i)};
            end
            %%{
            M = cat(2,M_c{:});
            
            [ut,~,~] = svd(M);
            Out = ut(:,J-R2+1:end);
    
            
    %% For A
elseif index==1
    
     IOhm = [];

        for i=1:J
             % index set 
             Ymis_i = Ymis(:,i,:);
             Ymis_i(isnan(Ymis_i)) = 0;
             %Ymis_i = reshape(Ymis_i, I, K);
             Ymis_i = squeeze(Ymis_i);
             %Di = get_Di(W,i);
             if rank(Ymis_i)==R1
                IOhm(i)=i; 
             end
        end     
        IOhm = IOhm(IOhm~=0); 
             
             % SIohm
             SIOhm= {};

             for i=1:numel(IOhm)
                 %SIOhm{i}=row_sel(reshape(Ymis(:,IOhm(i),:), I, K));
                 SIOhm{i}=row_sel(squeeze(Ymis(:,IOhm(i),:)));
             end

             % Yred
             %%{
             YredOhm ={};
             for i=1:numel(IOhm)
                 %Di = get_Di(W,IOhm(i));
                 Ymis_i = Ymis(:,IOhm(i),:);
                 Ymis_i(isnan(Ymis_i)) = 0;
                 %disp(size(Ymis_i))
                 %disp(size(C))
                 YredOhm{i}= squeeze(Ymis_i);
             end
             %out = cat(2,Yred{:});
             %}
            
            %YredOhm = squeeze(num2cell(Yred,[1 3]));
            % U_|_
             P = cellfun(@(x,y)x*y, SIOhm,YredOhm, 'UniformOutput',false);
            %[U, ~, ~] = cellfun(@(x) svd(x), P, 'UniformOutput', false);
            Us = {};
            for i =1:numel(P)
                Ip = nnz(diag(get_Di(W,IOhm(i))));
                [up, ~,~] = svd(P{i});
                Us{i}=up(:,1:Ip);
            end
            %U_c = cellfun(@(x) x(:, Ii-R2+1:end), U, 'UniformOutput', false);

            U_c = {};
            for i =1:numel(Us)
                Ii = nnz(diag(get_Di(W,IOhm(i))));
                U_c{i} = SIOhm{i}'*Us{i}(:,Ii-R1+1:end);
            end
            %%{
            M = cat(2,U_c{:});
            
            [ut,~,~] = svd(M);
            Out = ut(:,I-R1+1:end);
            %}
else
    disp('Please Specify the index')
end