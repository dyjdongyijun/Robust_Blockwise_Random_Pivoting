
clear all
close all

%load mnist.mat
load cifar10.mat


% randomly select 1000 images
B = A(:, randperm(60000, 1000));
%B = A;
X = B ./ vecnorm(B);


blk = 64;
ranks = blk * (6:6);


% SVD is optimal
[~, S, ~] = svd(X,"econ","vector");

err_svd = zeros(1,length(ranks));
for i=1:length(ranks)
    err_svd(i) = norm(S(ranks(i)+1:end)) / norm(S);
end
err_svd = err_svd.^2; % squared


%maxNumCompThreads(1);


%% CPQR
tic; 
[~, R1, p1] = qr(X,'econ','vector'); 
t_cpqr = toc;

res_cpqr = zeros(1,length(ranks));
for i=1:length(ranks)
    res_cpqr(i) = norm(R1(ranks(i)+1:end,ranks(i)+1:end),"fro") / norm(R1,"fro");
end
res_cpqr = res_cpqr.^2;

% time for interpolation matrix
err_cpqr = zeros(1,length(ranks));
for i=1:length(ranks)
    tic;
    T = R1(1:ranks(i),1:ranks(i)) \ R1(1:ranks(i),ranks(i)+1:end);
    t_cpqr_id(i) = toc;

    sk = p1(1:ranks(i));
    rd = p1(ranks(i)+1:end);
    err_cpqr(i) = norm(X(:,rd)-X(:,sk)*T, 'fro') / norm(X, 'fro');    
end
err_cpqr = err_cpqr.^2;



%% RBRP


tic
[Q2, R2, p2, res2] = RBRP(X, blk, 'random');
t_rbrp = toc;

res_rbrp = zeros(1,length(ranks));
for i=1:length(ranks)
    res_rbrp(i) = norm(R2(ranks(i)+1:end,ranks(i)+1:end),"fro") / norm(R2,"fro");
end
res_rbrp = res_rbrp.^2;

err_rbrp = zeros(1,length(ranks));
t_rbrp_id = zeros(1,length(ranks));
for i=1:length(ranks)
    tic;
    T = R2(1:ranks(i),1:ranks(i)) \ R2(1:ranks(i),ranks(i)+1:end);
    t_rbrp_id(i) = toc;

    sk = p2(1:ranks(i));
    rd = p2(ranks(i)+1:end);
    err_rbrp(i) = norm(X(:,rd)-X(:,sk)*T, 'fro') / norm(X, 'fro');
end
err_rbrp = err_rbrp.^2;




%% SqNorm


err_sqn = zeros(1,length(ranks));
time_sqn = zeros(1,length(ranks));
for i=1:length(ranks)
    tr = tic;
    [sk, rd, T] = SqNorm(X, ranks(i));
    time_sqn(i) = toc(tr);

    err_sqn(i) = norm(X(:,rd)-X(:,sk)*T, 'fro')^2 / norm(X,'fro')^2;
end



%% plot


res_dest = res2(1:length(ranks)) ./ norm(X, 'fro')^2;



figure(1)
plt=semilogy(ranks, err_svd, 'ok-',...
    ranks,res_cpqr,'r--',...
    ranks,res_rbrp,'g-',...
    ranks,err_sqn);


grid on
set(plt,'LineWidth',3,'MarkerSize',15);
ax = gca;
set(ax,'fontsize',22);
xlabel('rank');
ylabel('squared error');
legend('SVD', 'CPQR', 'RBRP', 'SqNorm')




% %%
% 
% figure(2)
% plt2=plot(ranks, time_cpqr, 'ro-',...
%     ranks,time_randLUPP, 'co-', ...
%     ranks,time_randLUPP_OS, 'bo-', ...
%     ranks,time_sqn, 'yo-', ...
%     rank_rbrp(1:m), time_rbrp(1:m)+t_rbrp_id(1:m), 'go-');
% 
% grid on
% set(plt2,'LineWidth',3,'MarkerSize',15);
% ax = gca;
% set(ax,'fontsize',22);
% xlabel('rank');
% ylabel('Runtime (sec)');
% legend('CPQR', 'randCPQR', 'randCPQR-OS', 'SqNorm', 'New','Location', 'Best')
% 

