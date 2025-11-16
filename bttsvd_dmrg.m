clear;
clc;

%% Reference

 % [1] Namgil Lee and Andrzej Cichocki. 2015. Estimating a Few Extreme Singular Values and Vectors
 %     for Large-Scale Matrices in Tensor Train Format. SIAM J. Matrix Anal. Appl. 36, 3 (2015), 994–1014.

%% Test example

% operator of which SVD is to be computed.
N = 7;
rA = [1 5*ones(1, N-1) 1]; % TT rank
In_row = [2*ones(1, N)]; 
In_col = [3*ones(1, N)];

% random tt-matrix
A  = tt_rand(In_row.*In_col, N, rA);
A = tt_matrix(A, In_row, In_col);

% defs
K=4; % number of singular values. 
nswp = 10; %maximal number of sweeps 
rmax = 100; %maximal rank 
tol_relres = 1e-10; 
delta = tol_relres/10/sqrt(N-1);
tol_grad = 1e-6; 
svds_niter = 100; 
verb = true; 
delloc = min(1, delta * 100); 

%% Initialize U & V

In_row = A.n; %row sizes of A
In_col = A.m; %col sizes of A

Ru = ceil(K/In_row(N)); 
Ru = [1; Ru*ones(N-1,1); K]; 
Ru(find(In_row==1)+1)=1;  %set ranks to 1; For example In_row = [1;1;2;2;2;2]


U = tt_rand(In_row, N, Ru, 1); 
Ru = U.r; 
Ru(N+1) = 1;  %!!!
Gu = []; 
for n=1:N
    Gu{n} = U{n}; 
end

Gu{N} = reshape(Gu{N},[Ru(N),In_row(N),1,K]);


Rv = ceil(K/In_col(N)); 
Rv = [1; Rv*ones(N-1,1); K]; 
Rv(find(In_col==1)+1)=1; % !!! set ranks to 1; For example In_col = [1;1;2;2;2;2]


V = tt_rand(In_col, N, Rv, 1); 
Rv = V.r; 
Rv(N+1) = 1; % !!!
Gv = []; 
for n=1:N
    Gv{n} = V{n}; 
end
Gv{N} = reshape(Gv{N},[Rv(N),In_col(N),1,K]);

%% Begin TT-ALS/DMRG-I
SV = diag(dot(U,round(A*V,tol_relres/10))); %to detect local convergence

% initialize left and right enviornments
PhiL = cell(1,N);
for k=1:N-1
   PhiL = left_interface(k, PhiL, A, Gu, Gv, rA, Ru, Rv, In_row, In_col);
end

PhiR = cell(1,N);
PhiR{N} = 1; 

%{
for k=N:-1:2
   PhiR = right_interface(k, PhiR, A, Gu, Gv, rA, Ru, Rv, In_row, In_col);
end
%}


% iteration
for swp=1:nswp
    if verb
        fprintf(['\n ---- ALS svd sweep ' num2str(swp) ' ----'])
    end
    flag_dec = 1; %right-to-left half sweep    
    SV_old = SV; 
    
    for n = [N:-1:2 1:N-1]
        if n==1
            flag_dec = 0; 
            delloc = delta; 
        end

        %SV_old = SV; 

        %%%%%%%%%% Estimate cores Wu & Wv of U & V %%%%%%%%%%

        %compute full matrix An=U(!=n)'AV(!=n)
        k=n;
        An  = effective_op(k, PhiL, PhiR, A, rA, Ru, Rv, In_row, In_col);
       
        %optimize (n)th TT-cores Wu & Wv
        [Wu,D,Wv] = svd(An); % eig
        D = diag(D);                    %D has size(An,1) singular values
        [D,idx] = sort(D,'descend');    % 
        K2 = min(size(An,2),min(size(An,1),K));
        SV = D(1:K2); 
        Wu = Wu(:,idx(1:K2)); 
        Wv = Wv(:,idx(1:K2)); 

        %%%%%%%%%% Matrix Factorization and Rank Estimation %%%%%%%%%%
        
        if flag_dec

            K2=size(Wu,2); 
            %
            Zu = Wu.'; 
            Zu = reshape(Zu,[K2*Ru(n), In_row(n)*Ru(n+1)]); 
            [uU1,uS1,uV1] = svd(Zu,'econ');
            uS1 = diag(uS1);
            unewR = max( min( my_chop2(uS1, delloc*norm(uS1)), rmax), 1);
            %
            Zv = Wv.'; 
            Zv = reshape(Zv,[K2*Rv(n),In_col(n)*Rv(n+1)]); 
            [vU1,vS1,vV1] = svd(Zv,'econ');
            vS1 = diag(vS1);
            vnewR = max( min( my_chop2(vS1, delloc*norm(vS1)), rmax), 1);
            %
            uU1 = uU1(:,1:unewR)*diag(uS1(1:unewR)); 
            uV1 = uV1(:,1:unewR); 
            uU1 = reshape(uU1,K2,numel(uU1)/K2).'; 
            uU1 = reshape(uU1, [Ru(n),unewR*K2]); 
            Gu{n-1} = reshape( reshape(Gu{n-1}, [numel(Gu{n-1})/Ru(n),Ru(n)]) * uU1 , [Ru(n-1),In_row(n-1),unewR,K2]); 
            Gu{n} = reshape(uV1', [unewR, In_row(n), Ru(n+1)]); 
            Ru(n) = unewR; 
    
            vU1 = vU1(:,1:vnewR)*diag(vS1(1:vnewR)); %R(n)*K2 x vnewR
            vV1 = vV1(:,1:vnewR); %In(n)*R(n+1) x vnewR
            vU1 = reshape(vU1,K2,numel(vU1)/K2).'; 
            vU1 = reshape(vU1, [Rv(n),vnewR*K2]); 
            Gv{n-1} = reshape( reshape(Gv{n-1}, [numel(Gv{n-1})/Rv(n),Rv(n)]) * vU1 , [Rv(n-1),In_col(n-1),vnewR,K2]); 
            Gv{n} = reshape(vV1', [vnewR, In_col(n), Rv(n+1)]); 
            Rv(n) = vnewR;             
            
            
            %update Phi_right{n} based on G{n} %%%%
            k=n; 
            PhiR = right_interface(k, PhiR, A, Gu, Gv, rA, Ru, Rv, In_row, In_col);
        else
            K2=size(Wu,2); 
            %
            Zu = reshape(Wu,[Ru(n)*In_row(n),Ru(n+1)*K2]);
            [uU1,uS1,uV1] = svd(Zu,'econ');
            uS1 = diag(uS1);
            unewR = max(min(my_chop2(uS1, delloc*norm(uS1)), rmax), 1);
            %
            Zv = reshape(Wv,[Rv(n)*In_col(n),Rv(n+1)*K2]);
            [vU1,vS1,vV1] = svd(Zv,'econ');
            vS1 = diag(vS1);
            vnewR = max(min(my_chop2(vS1, delloc*norm(vS1)), rmax), 1);
            %
            uU1 = uU1(:,1:unewR); %R(n)*In(n) x unewR
            uV1 = uV1(:,1:unewR)*diag(uS1(1:unewR)); %R(n+1)*K2 x unewR
            Gu{n} = reshape(uU1, [Ru(n), In_row(n), unewR]); 
            uV1 = reshape(uV1, [Ru(n+1),K2*unewR]); 
            Gu{n+1} = reshape( uV1' * reshape(Gu{n+1},[Ru(n+1),numel(Gu{n+1})/Ru(n+1)]) , [K2, unewR*In_row(n+1)*Ru(n+2)]);
            Gu{n+1} = reshape(Gu{n+1}.', [unewR,In_row(n+1),Ru(n+2),K2]); 
            Ru(n+1) = unewR;

            vU1 = vU1(:,1:vnewR); %R(n)*In(n) x vnewR
            vV1 = vV1(:,1:vnewR)*diag(vS1(1:vnewR)); %R(n+1)*K2 x vnewR
            Gv{n} = reshape(vU1, [Rv(n), In_col(n), vnewR]); 
            vV1 = reshape(vV1, [Rv(n+1),K2*vnewR]); 
            Gv{n+1} = reshape( vV1' * reshape(Gv{n+1},[Rv(n+1),numel(Gv{n+1})/Rv(n+1)]) , [K2, vnewR*In_col(n+1)*Rv(n+2)]);
            Gv{n+1} = reshape(Gv{n+1}.', [vnewR,In_col(n+1),Rv(n+2),K2]); 
            Rv(n+1) = vnewR;

            %update Phi_left{n} based on G{n} %%%%
            k = n; 
            PhiL = left_interface(k, PhiL, A, Gu, Gv, rA, Ru, Rv, In_row, In_col);
        end
                    
    end %end for n

    %%%%%%%%%% Update U & V %%%%%%%%%%

    U.r = Ru; U.r(N+1)=K;
    U.ps = cumsum([1; In_row.* U.r(1:N).* U.r(2:N+1)]);
    U.core = zeros(1,U.ps(end)-1); 
    for n=1:N
        if n<N
            U{n}=Gu{n};
        else
            U{n}=reshape(Gu{n},size(U{n})); 
        end
    end

    V.r = Rv; V.r(N+1)=K;
    V.ps = cumsum([1; In_col.* V.r(1:N).* V.r(2:N+1)]);
    V.core = zeros(1,V.ps(end)-1); 
    for n=1:N
        if n<N
            V{n}=Gv{n};
        else
            V{n}=reshape(Gv{n},size(V{n})); 
        end
    end

    %%%%%%%%%% Convergence criterion %%%%%%%%%%
    %print results
    if verb
        fprintf('singular values:')
        for k=1:min(2,K)
            fprintf([num2str(SV(k))  ',']); 
        end
        fprintf([' max_rank:' num2str(max(Ru)) ',' num2str(max(Rv))])
    end

    %local convergence
    if norm( sort(SV_old) - sort(SV),'fro') < tol_grad * norm(SV_old,'fro')
        if verb
            fprintf('\nA local convergence at sweep %d\n',swp);
        end
        break; 
    end

end% end of swp

% display results


% - bttsvd
disp('via DMRG-SVD')
relres = norm( round(transpose(A)*U,tol_relres/10) - V*diag(SV))/norm(SV,'fro'); 
disp(['Relative Residual [tt]: ' num2str(relres)])
disp('Singular values [tt]: ')
disp(SV')

% - standard SVD
Afull = full(A);
[uf,sf,vf] = svds(Afull, K);

disp('via standard SVD')
relres = norm(Afull'*uf - vf*sf)/norm(sf,'fro'); 
disp(['Relative Residual [dense]: ' num2str(relres)])
disp('Singular values [dense]: ')
disp(diag(sf)')



%% Utility functions
function Phi_left = left_interface(k, Phi_left, A, Gu, Gv, rA, Ru, Rv, In_row, In_col)
   if k==1
       Phi_left{k} = 1;
   end
   %product from Guk
    Guk = reshape(Gu{k}, [Ru(k), In_row(k)*Ru(k+1)]); 
    Bleft = Guk' * Phi_left{k};  

    Bleft = reshape(Bleft, [In_row(k), Ru(k+1), rA(k), Rv(k)]); 
    Bleft = permute(Bleft, [2,4,3,1]); 
    Bleft = reshape(Bleft, [Ru(k+1)*Rv(k), rA(k)*In_row(k)]); 

    %product from A{k}
    Bleft = Bleft * reshape(A{k}, [rA(k)*In_row(k), In_col(k)*rA(k+1)]);

    Bleft = reshape(Bleft, [Ru(k+1), Rv(k)*In_col(k), rA(k+1)]); 
    Bleft = permute(Bleft, [1,3,2]); 
    Bleft = reshape(Bleft, [Ru(k+1)*rA(k+1), Rv(k)*In_col(k)]); 

    %products from Gvk
    Gvk = reshape(Gv{k}, [Rv(k)*In_col(k), Rv(k+1)]); 
    Bleft = Bleft * Gvk; 

    Phi_left{k+1} = reshape(Bleft, [Ru(k+1), rA(k+1)*Rv(k+1)]);  %!!!   
end

function Phi_right = right_interface(k, Phi_right, A, Gu, Gv, rA, Ru, Rv, In_row, In_col)
   if k==A.d
       Phi_right{k} = 1;
   end
    %product from Guk
    Guk = reshape(Gu{k}, [Ru(k)*In_row(k), Ru(k+1)]); 
    Bright = Guk * Phi_right{k}; 
    
    Bright = reshape(Bright, [Ru(k), In_row(k), rA(k+1), Rv(k+1)]); 
    Bright = permute(Bright, [1,4,2,3]); 
    Bright = reshape(Bright, [Ru(k)*Rv(k+1), In_row(k)*rA(k+1)]); 

    %products from A{k}
    Bright = Bright * reshape( permute(A{k},[2,4,1,3]), ...
                               [In_row(k)*rA(k+1), rA(k)*In_col(k)]); 
    
    Bright = reshape(Bright, [Ru(k), Rv(k+1), rA(k), In_col(k)]); 
    Bright = permute(Bright, [1,3,4,2]); 
    Bright = reshape(Bright, [Ru(k)*rA(k), In_col(k)*Rv(k+1)]); 
    
    %product from Gvk
    Gvk = reshape(Gv{k}, [Rv(k), In_col(k)*Rv(k+1)]); 
    Bright = Bright * Gvk'; 

    Phi_right{k-1} = reshape(Bright, [Ru(k), rA(k)*Rv(k)]);
end

function Ak  = effective_op(k, Phi_left, Phi_right, A, rA, Ru, Rv, In_row, In_col)

    %compute full matrix Ak=U(!=k)'AV(!=k)
    Ak = reshape(Phi_left{k}, [Ru(k)*rA(k),Rv(k)]).'; 
    Ak = reshape(Ak, [Rv(k)*Ru(k),rA(k)]); 
    Ak = Ak * reshape(A{k},[rA(k),numel(A{k})/rA(k)]); %Rv(k),Ru(k),In_row(k),In_col(k),rA(k+1)
    Ak = reshape(Ak,[numel(Ak)/rA(k+1),rA(k+1)]); 
    Ak = Ak * reshape(Phi_right{k}.',[rA(k+1),Rv(k+1)*Ru(k+1)]); 

    Ak = reshape(Ak, [Rv(k),Ru(k)*In_row(k),In_col(k)*Rv(k+1),Ru(k+1)]); 
    Ak = permute(Ak,[2,4,1,3]); 
    Ak = reshape(Ak, [Ru(k)*In_row(k)*Ru(k+1),Rv(k)*In_col(k)*Rv(k+1)]);

end
