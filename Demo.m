% run this demo.
clear
clc
warning off;

name = 'PaviaU';
load(sprintf('dataset_%s.mat', name));

[B,~] = size(data); % number of trained samples, band dimension
param.B = B; param.tol = 0.001; param.maxIter = 10^2; param.p = 2; param.flag=1;
param.lambda = 0.1; param.beta = 10^-7; param.alpha = 10^-7; param.r = 2.5;
%% main loop
Xtrain = data(:,trainIdx);
Ytrain = labels(trainIdx);
Xtest = data(:,testIdx);
Ytest = labels(testIdx);
% set the label matrix Ymatrix for Ytrain
Ymatrix = -1 .* ones(max(Ytrain), length(Ytrain));
for i = 1:length(Ytrain)
    Ymatrix(Ytrain(i),i) = 1;
end
fprintf('LvaHAl start!.\n');
fprintf('... generating different views.\n');
tstart = tic;
% generating different views
tauMin_B = 1/3; tauMax_B = 1/2; flag_B = zeros(B,1);
totalCount = 1; count = 0;
bandsIn = cell(totalCount,1);
while true
    if sum(flag_B) == B && count >= totalCount
        break; 
    end
    count = count + 1;
    tau_B = tauMin_B + (tauMax_B - tauMin_B) * rand(1);
    bandsIn{count} = zeros(floor(tau_B*B),1);
    count1 = 0;
    while sum(bandsIn{count} == 0) ~= 0
        j = floor(1+B*rand(1));
        if sum(bandsIn{count} == j) == 0
            count1 = count1 + 1;
            bandsIn{count}(count1) = j;
        end
    end
    bandsIn{count} = sort(bandsIn{count});
    flag_B(bandsIn{count}) = 1;
end
Xv = cell(length(bandsIn),1);
for i = 1:length(bandsIn)
    Xv{i} = Xtrain(bandsIn{i},:);
end
%% learning model and selecting the expected bands
fprintf('... band selection.\n');
Wopt = zeros(B, max(Ytrain));
[Wopt, Wv, bv, obj] = LvaHalAlg(Xv, Ymatrix, Wopt, bandsIn, param);
telasped = toc(tstart);
fprintf(sprintf('.... LvaHAl end£¡The consumed tims %d(s).\n',telasped));
[qY, qI] = sort(sqrt(sum(Wopt.^2,2)),'descend');
bandNum = 24;
selectedBands = sort(qI(1:bandNum));
%% training classifier and perform classification
xtrain2 = Xtrain(selectedBands,:); ytrain = Ytrain;
xtest2 = Xtest(selectedBands,:); ytest = Ytest;
optimalNumNeighbors = 0;
optimalLoss = 10^2;
kArray = [2, 3, 5, 7, 10, 15];  % the value of K for KNN
for kId = 1:length(kArray)
    mdl = ClassificationKNN.fit(xtrain2', ytrain, 'NumNeighbors', kArray(kId));
    cvmdl = crossval(mdl, 'kfold', 5);
    cvmdlloss = kfoldLoss(cvmdl);
    if cvmdlloss < optimalLoss
        optimalLoss = cvmdlloss;
        optimalNumNeighbors = kArray(kId);
    end
end
mdl = ClassificationKNN.fit(xtrain2', ytrain, 'NumNeighbors', optimalNumNeighbors);
tlabs = predict(mdl, xtest2');
[kappa, acc, acc_O, acc_A] = evaluate_results(tlabs, ytest);
fprintf('KNN: numbef of bands: %d, kappa: %d, OA: %d, AA: %d.\n', bandNum, kappa, acc_O, acc_A);