function [G_ortho] = Left_Right_Ortho(G, mu)
[ULeft]  = Left_Orth(G, mu);
[G_ortho]  = Right_Orth(ULeft, mu);
end