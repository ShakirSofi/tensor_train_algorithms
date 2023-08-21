function Gm = give_mid_cr(Tmred, GL)
% GL: ... x rn-1; GR: rn x ...   ; Tmiss:  I_1 x I_2 x r_n2
r_n1 = size(GL,2);  I_n = size(Tmred,2);  r_n2 = size(Tmred,3); 
Gm = zeros(r_n1, I_n, r_n2);

for ii=1:I_n
    slice = squeeze(Tmred(:,ii,:));
    Gm(:,ii,:) = sol_slice(slice, GL);
end

%%%%%
function [Gms] = sol_slice(Ts, GL)
 xout = Solve_ortho(GL, Ts);
 %missingIndices = any(isnan(Ts), 2);
 %Ts(missingIndices,:) = xout(size(GL,2)+1:end,:);
 Gms =  xout(1:size(GL,2),:); %GL.'*Ts;
end
%%%%
end