
function [Q, R, perm, res, time, rank] = RBRP(A, blk, mode)

    if isgpuarray(A)
      fprintf('RBRP on GPU')
    else
      fprintf('RBRP on CPU with %d threads\n', maxNumCompThreads)
    end

[m, n] = size(A);
perm = 1:n;

diags = vecnorm(A).^2;
res = [];
time = [];
rank = [];
nrmA = norm(A, 'fro');

p = min(m,n);

if isgpuarray(A)
  Q = zeros(m, p, 'gpuArray');
  R = zeros(p, n, 'gpuArray');
else
  Q = zeros(m, p);
  R = zeros(p, n);
end

i = 1;
while i<p+1
    t_rbrp=tic;
    bs = min([blk, p-i+1, sum(diags(i:n)>0)]);

    % sampling a subset of indices
    if strcmp(mode, 'greedy') == 1
        % Greedy
        [~, ind] = sort(diags(i:n), 'descend');
        J  = ind(1:bs) + i-1; 
    else
        % Sampling
        w = diags(i:n);
        %w = sqrt(diags(i:n));
        %prob = w/sum(w);
        %J  = unique(randsmpl(prob,bs,1)) +i-1;

        if isgpuarray(A)
            w = gather(w);
            bs = gather(bs);
        end
        J = datasample(i:n, bs, 'Replace', false, 'Weights', w);
    end

    % CPQR on the panel
    X = A(:,J);

    % Gram Schmidt
    I1 = 1:i-1;
    X = X - Q(:,I1) * R(I1,J);
    %X = X - Q(:,I1) * (Q(:,I1)' * X);

    [Q2, R2, p2] = qr(X, 'econ', 'vector');
    J = J(p2); % reorder selected indices

    % filter
    v = sum(R2.^2, 2);
    v = cumsum(flip(v));
    bs = sum(v > v(end)/blk);


    %bs = length(J);
    I2 = i:i+bs-1;
    I3 = i+bs:n;


    % update outputs
    J = J(1:bs);
    Q(:,I2)  = Q2(:,1:bs);
    R(I2,I2) = R2(1:bs,1:bs);


    K1 = setdiff(I2, J, 'stable'); % indices out
    K2 = setdiff(J, I2, 'stable'); % indices in

    %K1 = J;
    %K2 = I2;


    % swamp columns of A
    T = A(:,K1);
    A(:,K1) = A(:,K2);
    A(:,K2) = T;

    % swamp columns of R
    T = R(I1,K1);
    R(I1,K1) = R(I1,K2);
    R(I1,K2) = T;

    % swamp columns of diags
    T = diags(K1);
    diags(K1) = diags(K2);
    diags(K2) = T;

    % swamp columns of perm
    T = perm(K1);
    perm(K1) = perm(K2);
    perm(K2) = T;


    % post processing
    R(I2,I3)  = Q2(:,1:bs)' * A(:,I3);
    diags(I3) = diags(I3) - sum(R(I2,I3).^2, 1);
    tmp = toc(t_rbrp);

    time = [time, tmp];
    res  = [res, sum(diags(I3))];
    rank = [rank, bs];

    % tiny negative
    diags(diags<0) = 0;
    if sum(diags(I3)>0) == 0
        fprintf('zero residual\n')
        break
    end


    % next block
    i = i + bs;
end

res = res ./ nrmA^2;
time = cumsum(time);
rank = cumsum(rank);
end
