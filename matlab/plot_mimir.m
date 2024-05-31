

results_mimir_gpu


%%


ranks = T1(1,:);
err_svd = T1(2,:);
res_cpqr = T1(3,:);
err_randCPQR = T1(4,:);
err_randCPQR_OS = T1(5,:);
err_randLUPP = T1(6,:);
err_randLUPP_OS = T1(7,:);
err_sqn = T1(8,:);

time_cpqr = T1(9,:);
time_randCPQR = T1(10,:);
time_randCPQR_OS = T1(11,:);
time_randLUPP = T1(12,:);
time_randLUPP_OS = T1(13,:);
time_sqn = T1(14,:);



rank_rbrp = T2(1,:);
res_rbrp = T2(2,:);
time_rbrp = T2(3,:);



m = sum(rank_rbrp < ranks(end))+1;


figure(1)
plt=semilogy(ranks, err_svd, 'ok-',...
    ranks,res_cpqr,'ro-',...
    ranks,err_randLUPP, 'co-', ...
    ranks,err_randLUPP_OS, 'bo-', ...
    ranks,err_sqn, 'yo-', ...
    rank_rbrp(1:m),res_rbrp(1:m),'go-');


grid on
set(plt,'LineWidth',3,'MarkerSize',15);
ax = gca;
set(ax,'fontsize',22);
xlabel('rank');
ylabel('squared error');
legend('SVD', 'CPQR', 'randCPQR', 'randCPQR-OS', 'SqNorm', 'New','Location', 'southwest')


%%

figure(2)
plt2=plot(ranks, time_cpqr, 'ro-',...
    ranks,time_randLUPP, 'co-', ...
    ranks,time_randLUPP_OS, 'bo-', ...
    ranks,time_sqn, 'yo-', ...
    rank_rbrp(1:m), time_rbrp(1:m), 'go-');

grid on
set(plt2,'LineWidth',3,'MarkerSize',15);
ax = gca;
set(ax,'fontsize',22);
xlabel('rank');
ylabel('Runtime (sec)');
legend('CPQR', 'randCPQR', 'randCPQR-OS', 'SqNorm', 'New','Location', 'Best')
