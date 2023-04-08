function [X] = rep0(X)
X(isnan(X))=0;
end