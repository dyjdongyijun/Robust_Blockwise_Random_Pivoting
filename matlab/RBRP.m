
function [Q, R, perm, res, time] = RBRP(A, blk, mode)

[m, n] = size(A);
perm = 1:n;

diags = vecnorm(A).^2;
res = [];
time = [];
nrmA = norm(A, 'fro');

p = min(m,n);
Q = zeros(m, p);
R = zeros(p,n);

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
        w = sqrt(diags(i:n));
        %prob = w/sum(w);
        %J  = unique(randsmpl(prob,bs,1)) +i-1;

        w = gather(w);
        bs = gather(bs);
        J = datasample(i:n, bs, 'Replace', false, 'Weights', w);
    end

    %J  = unique(J);
    %K  = setdiff(i:n, J, 'sorted');


    % CPQR on the panel
    X = A(:,J);    
    [Q2, R2, p2] = qr(X, 'econ', 'vector');
    J = J(p2); % reorder selected indices


    bs = length(J);
    I1 = 1:i-1;
    I2 = i:i+bs-1;
    I3 = i+bs:n;

    % update outputs
    Q(:,I2) = Q2;
    R(I2,I2) = R2;


    %K1 = setdiff(I2, J,'stable'); % indices out
    %K2 = setdiff(J, I2,'stable'); % indices in

    K1 = J;
    K2 = I2;

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
    R(I2,I3)  = Q2' * A(:,I3);
    A(:,I3)   = A(:,I3) - Q2 * R(I2,I3);
    diags(I3) = diags(I3) - sum(R(I2,I3).^2, 1);
    res = [res, sum(diags(I3))];

    % tiny negative
    diags(diags<0) = 0;
    if sum(diags(I3)>0) == 0
        break
    end


    % next block
    i = i + bs;
    tmp = toc(t_rbrp);
    time = [time, tmp];
end

res = res ./ nrmA^2;
time = cumsum(time);
end
