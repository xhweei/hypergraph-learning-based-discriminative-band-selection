% Evaluate Classification Results
%
% Syntax:
%       [kappa acc acc_O acc_A] = evaluate_results(tlabs, Trlabs)
%
% Input:
%       tlabs:  M-by-1 vector of labels given to test data
%       Trlabs: M-by-1 vector of ground truth labels
%
% Output:
%       kappa:  kappa coefficient
%       acc:    accuracy per class
%       acc_O:  overall accuracy
%       acc_A:  average accuracy
% 
% Written by Ali Soltani-Farani <a_soltani@ce.sharif.edu>
% Copyright 2012 by Ali Soltani-Farani

function [kappa acc acc_O acc_A] = evaluate_results(tlabs, Trlabs)

c = max(Trlabs) - min(Trlabs) + 1;

% make confusion matrix

CM = zeros(c,c);

for i = 1:c
    for j = 1:c
        CM(i,j) = sum(tlabs==i & Trlabs==j);
    end
end

% Class accuracy
acc = zeros(c, 1);
for j = 1:c
    acc(j) = CM(j,j)/sum(CM(:,j));
end

% Overall and average accuracy
acc_O = sum(diag(CM))/sum(sum(CM));
acc_A = mean( acc );

% Kappa coefficient of agreement
% Incorrect: kappa = (acc_O - sum( sum(CM,2).^2 )/(sum(sum(CM))^2))/(1-sum( sum(CM,2).^2 )/(sum(sum(CM))^2));
kappa = (acc_O - sum( sum(CM,1)*sum(CM,2) )/sum(sum(CM)).^2)/(1-sum( sum(CM,1)*sum(CM,2) )/sum(sum(CM)).^2);
end
