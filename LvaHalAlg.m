function [Wopt, Wv, bv, obj] = LvaHalAlg(Xv, Y, Wopt, bandsIn, param)
% Input parameters:
%   Xv      -       Xv is a v-dimensional cell. Each cell denotes a dv*n
%   sample matrix.
%   Y       -       Y is a c*n label matrix. Y must be 1 or -1.
%   Wopt    -       Wopt is a d*c optimal weight matrix. d = \sum_v {dv}.
%   bandsIn -       bandsIn is a v-dimensional cell. Each cell denotes a
%   dv-dimensional vector including the indexes of bands for each view.
%   param   -
%       lambda: the regularization parameter lambda.
%       beta:   the regularization parameter beta.
%       alpha:  the regularization parameter alpha for each view.
%       r:      the regularization parameter r.
%       p:      the l_p norm. Default value is 2.
%       tol:    the tolerence.
%       maxIter:the maximum number of iterations.
%       flag:   decide the number of iterations of solving Wv, bv, and Mv.
% Output parameters:
%  Wv       -       Wv is a v-dimensional cell. Each cell denotes a dv*c
%  weight matrix.
%  bv       -       bv is a v-dimensional cell. Each cell denotes a
%  c-dimensional vector.
%  obj      -       obj is a vector including the value of objective
%  function for each iteration.
%% initial fixed parameters
lambda = param.lambda; beta = param.beta; alpha = param.alpha; r = param.r; 
p = param.p; tol = param.tol; maxIter = param.maxIter; flag = param.flag;
[c,n] = size(Y); v = length(bandsIn); d = size(Wopt,1);

Wv = cell(v,1); bv = cell(v,1); Mv = cell(v,1); gv = cell(v,1);
epsilon = cell(v,1); lossFun = cell(v,1);
%% processing hypergraph
Hi = zeros(n, c);  % the incident matrix. 该变量在整个优化过程中及所有views是固定的
De = zeros(c,1); % 超边度: 该变量在整个优化过程中是固定的.
for i = 1:c
    Hi(Y(i,:)==1,i) = 1; % Hi(j,i)=1表示第i个样本属于第j个类
    De(i) = sum(Hi(:,i));
end
Hw = cell(v,1); % 超边权值: 该变量是需要学习的. 凡与该变量相关的变量均需要更新
Da = cell(v,1); % 顶点度：该向量需要根据Hw的变化而调整
for vId = 1:v
    Hw{vId} = 1/c .* ones(c,1); % 为每一个局部视角：初始化超边权值
    Da{vId} = Hi * Hw{vId}; % 为每一个局部视角：计算所有顶点的度
end
%% processing matrix Cmatrix
Cmatrix = zeros(d,d);
for vId = 1:v
    Ctmp = zeros(d,1); Ctmp(bandsIn{vId}) = 1; Ctmp = diag(Ctmp);
    Cmatrix = Cmatrix + alpha * (Ctmp' * Ctmp);
end
%% main program
if flag == 1; inMaxIter = maxIter; else; inMaxIter = 1; end
obj = zeros(maxIter,1);
for iter = 1:maxIter
    %% optimize Wv, bv, Mv for each view
    for vId = 1:v
        X = Xv{vId}; dim = size(X,1);  % dim = length(bandsIn{vId}); X \in R^{dv \times n}
        L = diag(Da{vId}) - Hi * diag(Hw{vId}) * diag(1./De) * Hi';
        inObj = zeros(maxIter,1);
        for inIter = 1:inMaxIter
            % initialize
            if iter == 1; Wv{vId} = zeros(dim,c); Mv{vId} = zeros(c,n); gv{vId} = ones(n,1); end
            P = diag(1 ./ (2 * sqrt(sum(Wv{vId}.^2,2)) + 10^-6));
            Wsub = Wopt(bandsIn{vId},:);
            Z = Y + Y .* Mv{vId};  % Z \in R^{c \times n}
            H = diag(gv{vId}) - 1/sum(gv{vId}) * gv{vId} * gv{vId}';
            % calculate Wv
            Wv{vId} = (X*H*X' + lambda*P + 2*beta*X*L*X' + alpha*eye(dim)) \ (X*H*Z' + alpha*Wsub);
            % calculate bv
            bv{vId} = 1/sum(gv{vId}) * (Z*gv{vId} - Wv{vId}'*X*gv{vId});  % bv \in R^{c}
            % calculate Mv
            E1 = Wv{vId}'*X + bv{vId} * ones(1,n) - Y;
            Mv{vId} = max(Y.*E1, 0);
            % update gv
            E = Wv{vId}'*X + bv{vId}*ones(1,n) - Y - Y.*Mv{vId};
            dee = sqrt(sum(E.*E)+eps);
            if iter > 4
                idxe = find(dee.^p < epsilon{vId});
            else
                [temp, ideff] = sort(dee.^p);
                numeff = ceil(0.9*n);
                idxe = ideff(1:numeff);
                epsilon{vId} = temp(numeff+1);
            end
            gv{vId} = zeros(n,1);
            gv{vId}(idxe) = 0.5*p*dee(idxe).^(p-2);
            lossFun{vId} = sum(dee(idxe).^p) + epsilon{vId}*(n-length(idxe));
            inObj(inIter) = lossFun{vId} + lambda * sum(sqrt(sum(Wv{vId}.^2, 2))) + ...
                alpha * norm(Wv{vId} - Wsub, 'fro')^2 + ...
                2 * beta * trace(Wv{vId}' * Xv{vId} * L * Xv{vId}' * Wv{vId});
            if inIter > 1
                err = abs(inObj(inIter) - inObj(inIter-1)) / abs(inObj(inIter));
%                 fprintf('inIter = %d, inObj = %d, err = %d.\n', inIter, inObj(inIter), err);
                if err < tol; break; end
            else
%                 fprintf('inIter = %d, inObj = %d.\n', inIter, inObj(inIter));
            end
        end
    end
    %% optimize hypergraph
    for vId = 1:v
        upsilon = zeros(c,1);
        betavId = zeros(n,n);
        for i = 1:n
            tmpBeta = bsxfun(@minus, repmat(Xv{vId}(:,i), 1, De(Y(:,i)==1)), Xv{vId}(:, Y(Y(:,i)==1,:)==1));
            betavId(Y(Y(:,i)==1,:)==1, i) = sum((Wv{vId}' * tmpBeta).^2)';
        end
        for cId = 1:c
            tmp = betavId(Y(cId,:)==1, Y(cId,:)==1);
            upsilon(cId) = sum(sum(tmp)) / De(cId);
        end
        Hw{vId} = (upsilon.^(1/(1-r))) ./ sum(upsilon.^(1/(1-r)));
        Hw{vId} = Hw{vId} .^ r;
        Da{vId} = Hi * Hw{vId};
    end
    %% optimal Wopt
    Wmatrix = zeros(size(Wopt));
    for vId = 1:v
        Wtmp = zeros(size(Wopt));
        Wtmp(bandsIn{vId},:) = Wv{vId};
        Ctmp = zeros(d,1); Ctmp(bandsIn{vId}) = 1; Ctmp = diag(Ctmp);
        Wmatrix = Wmatrix + alpha * (Ctmp' * Wtmp);
    end
    Wopt = Cmatrix \ Wmatrix;
    %% calculate the value of objective function
    for vId = 1:v
        obj(iter) = obj(iter) + lossFun{vId} + lambda * sum(sqrt(sum(Wv{vId}.^2, 2))) + ...
            alpha * norm(Wv{vId} - Wopt(bandsIn{vId},:), 'fro')^2;
        L = diag(Da{vId}) - Hi * diag(Hw{vId}) * diag(1./De) * Hi';
        obj(iter) = obj(iter) + 2 * beta * trace(Wv{vId}' * Xv{vId} * L * Xv{vId}' * Wv{vId});
    end
    if iter > 1
        err = abs(obj(iter) - obj(iter-1)) / abs(obj(iter));
%         fprintf('iter = %d, obj = %d, err = %d.\n', iter, obj(iter), err);
        if err < tol; break; end
    else
%         fprintf('iter = %d, obj = %d.\n', iter, obj(iter));
    end
end

end

