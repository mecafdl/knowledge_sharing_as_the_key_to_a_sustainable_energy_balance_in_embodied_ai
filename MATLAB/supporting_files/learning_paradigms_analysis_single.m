close all
alpha = 0.5;
figure
N_tot = 10;
Nj = 0:N_tot;
% plot(Nj,exp(-alpha*Nj) - exp(-alpha*10),'o-')
plot(Nj,1-(1/(1 - exp(-alpha*N_tot)))*(exp(-alpha*Nj) - exp(-alpha*N_tot)),'o-')
hold on
plot(Nj,Nj/Nj(end),'ro-')
grid minor

%% ************************************************************************
%                      Single incremental learning                        *
% *************************************************************************
clc
close all
% clf
N_S   = 10;
N_K   = 1;
% N_S   = 10;
% N_K   = 5;
alpha = 0.1;
% -------------------------------------------------------------------------
figure('Color',[1 1 1])
C = 0;
cj = zeros(N_S/N_K,1); % complexity for each of the skills in a cluster
sigma_bar = zeros(N_S/N_K,1); 
for j = 1:N_S/N_K % loop over the skills in a cluster
    sigma_bar(j) = exp(-alpha*(j-1));
    cj(j)        = c0*sigma_bar(j);
    n            = 0:0.001:cj(j); 
%     area(n + 0*C,exp(-5/cj*n),'FaceColor','flat')
%     p = area(n + 0*C, 1*exp(-6/cj(j)*n),'FaceAlpha',0.5);
%     p = area(n + 1*C,  exp(-alpha*(j))*(1 -exp(-6/cj(j)*n)) + (1-exp(-alpha*(j-1))),'FaceAlpha',0.5);
    p = area(n + 1*C,  (sigma_bar(j) - exp(-alpha*(j)))*(exp(-6/cj(j)*n)) + 1*exp(-alpha*(j)),'FaceAlpha',0.5);
    hold on
    C = C + cj(j);
end
% plot(cumsum(cj),sigma_bar,'k--')
plot([0;cumsum(cj)],[sigma_bar;exp(-alpha*(N_S/N_K))],'ro--')

xlabel('$n$','Interpreter','latex','FontSize',25);
ylabel('$\bar{\sigma}_{j,k}^{(I)}$','Interpreter','latex','FontSize',25);   
title('Remaining Knowledge VS. Episode (Single Incremental)','FontSize',15)
grid on
% Knowledge --------------------------------------------------------------
figure('Color',[1 1 1])
plot(1:N_S/N_K, 1-sigma_bar,'-or','LineWidth', 2)
xlabel('$N_{j}$','Interpreter','latex','FontSize',20);
ylabel('${\bar{\sigma}}_{j}^{(I)}$','Interpreter','latex','FontSize',20); 
xlim([1 N_S/N_K])
title('Remaining Knowledge VS. Skills (Single Incremental)','FontSize',15)
grid on
% Complexity --------------------------------------------------------------
figure('Color',[1 1 1])
plot(1:N_S/N_K,cj,'-o','LineWidth', 2)
xlabel('$N_{j}$','Interpreter','latex','FontSize',20);
ylabel('$c_{j}^{(I)}$','Interpreter','latex','FontSize',20); 
xlim([1 N_S/N_K])
title('Complexity VS. Skills (Single Incremental)','FontSize',15)
grid on
% -------------------------------------------------------------------------
figure('Color',[1 1 1])
C = 0;
cj = zeros(N_S/N_K,1);
sigma_bar = zeros(N_S/N_K,1);
for j = 1:N_S/N_K
    sigma_bar(j) = exp(-alpha*(j-1));
    cj(j)        = c0*sigma_bar(j);
    n            = 0:0.1:cj(j);
    p1            = plot3(n, j*ones(numel(n),1), sigma_bar(j)*exp(-6/cj(j)*n),'LineWidth',2);
    hold on
    p            = fill3(n([1 1:end end]), j*ones(numel(n)+2,1), [0 sigma_bar(j)*exp(-6/cj(j)*n) 0],p1.Color, 'FaceAlpha', 0.1);
    C            = C + cj(j);
end
plot3(zeros(N_S/N_K,1),1:N_S/N_K,sigma_bar ,'d--','LineWidth', 2,'color',[1 0 0])
plot3(cj,1:N_S/N_K,zeros(N_S/N_K,1),'o--','LineWidth', 2,'color',[0 0 1])
xlabel('$n$','Interpreter','latex','FontSize',20);
ylabel('$N_{j}$','Interpreter','latex','FontSize',20);   
zlabel('$\bar{\sigma}_{j,k}^{(I)}$','Interpreter','latex','FontSize',20);   
ylim([1 N_S/N_K])
view([150 30])
SAVE_FIG = 0;
fig = gcf;
ax  = gca;
% plt = p;
% fcn_scrpt_prepare_graph(fig, ax, [], [], [], 8.8, 4, 1)
% axis off
view([45,30])

if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','single_incremental_knowledge'),'-pdf')
    close(gcf);
end      

%% ************************************************************************
%                      Single transfer learning                           *
% *************************************************************************
close all
clc
N_S   = 60;
N_K   = 4;
alpha = 0.2;
% -------------------------------------------------------------------------
figure('Color',[1 1 1])
hold on
clear sigma_bar
C            = 0;
sigma_k      = zeros(N_K,1);
bar_sigma_jk = zeros(N_S/N_K,N_K);
cjk          = zeros(N_S/N_K,N_K);
transfer     = NaN(N_K,1);
for k = 1:N_K
    if k == 1
        transfer(k) = 1;
    else
        %beta        = 1/(k-1);
        beta        = 1/(N_K-1);
        transfer(k) = 1 - beta*sum(sigma_k(1:k-1));
    end
    plot(k*c0*ones(11,1),0:0.1:1,'k-','LineWidth', 0.5)
    for j = 1:N_S/N_K
        bar_sigma_jk(j,k) = transfer(k)*exp(-alpha*(j-1));
        cjk(j,k)          = c0*bar_sigma_jk(j,k);
        n                 = 0:0.1:cjk(j,k);
        p                 = plot(n + (k-1)*c0,bar_sigma_jk(j,k)*exp(-6/cjk(j,k)*n),'LineWidth',1);
        p                 = area(n + (k-1)*c0,bar_sigma_jk(j,k)*exp(-6/cjk(j,k)*n),'EdgeColor',p.Color,'FaceColor',p.Color,'FaceAlpha',0.1);
        C                 = C + cjk(j,k);
    end
    ax = gca;
    ax.ColorOrderIndex = 1;
    sigma_k(k) = 1 - bar_sigma_jk(j,k);
end
plot(c0*(0:1:N_K-1),transfer,'ks--','LineWidth',2)
xlabel('$n$','Interpreter','latex','FontSize',20);
ylabel('$\bar{\sigma}_{j,k}^{(T)}$','Interpreter','latex','FontSize',20);   
% title('Knowledge VS. Episode (Single Transfer)','FontSize',15)
grid on
SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, [], [], [], 8.8, 4, 1)
if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','single_transfer_knowledge'),'-pdf')
    close(gcf);
end    
% -------------------------------------------------------------------------
figure('Color',[1 1 1])
plot(1:N_S/N_K,cjk,'-o','LineWidth', 2)
xlabel('$N_{j,k}$','Interpreter','latex','FontSize',20);
ylabel('$c_{j,k}^{(T)}$','Interpreter','latex','FontSize',20);
legend('k_1','k_2','k_3','k_4','k_5')
xlim([1 N_S/N_K])
% title('Complexity VS. Skills (Single Transfer)','FontSize',15)
grid on
SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, [], [], [], 8.8, 4, 1)
if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','single_transfer_complexity'),'-pdf')
    close(gcf);
end    

% %% Single transfer learning
% 
% for i = 1:numel(N_K)
%     for j=1:numel(m)
%         bar_sigma_m(i,j) = exp(-alpha*N_S/(m(j)*N_K(i)));
%         factor(i,j)      = 1 - 0.5*(1 + N_K(i))*beta(i)*(1 - bar_sigma_m(i,j));
%         C_i_p(i,j)       = m(j)*N_K(i)*c0*(1 - exp(-alpha*N_S/(m(j)*N_K(i))))/(1 - exp(-alpha));
%         C_t_p(i,j)       = factor(i,j).*C_i_p(i,j);
%     end
% end
% C_c_p = m.*(c0*(1 - exp(-alpha*N_S))./(1 - exp(-alpha*m)));
% 
% close all
% N_S   = 10;
% alpha = 0.1;
% figure('Color',[1 1 1])
% hold on
% C = 0;
% for i = 1:N_S
%     cj = c0*exp(-alpha*i);
%     n = 0:0.1:cj;
% %     area(n + 0*C,exp(-5/cj*n),'FaceColor','flat')
%     p = area(n + 0*C,exp(-5/cj*n),'FaceAlpha',0.5);
%     C = C + cj;
% end
% xlabel('$n$','Interpreter','latex','FontSize',20);
% ylabel('$c^{(I)}$','Interpreter','latex','FontSize',20);   
% title('Incremental','FontSize',15)
% grid minor









%% ************************************************************************
%                           Parallel learning
% *************************************************************************
clc
N_K = 10;
N_S = 1:1:1E6;
alpha = 0.1;
beta = 1/N_K;
m = 10:10:100;
% m = [10 100];
c0 = 100;
% Incremental -------------------------------------------------------------
close all
figure('Color',[1 1 1])
p = [];
for i = 1:numel(m)
    C_i_p = m(i)*N_K*c0*(1 - exp(-alpha*N_S/(m(i)*N_K)))/(1 - exp(-alpha));
    p = [p loglog(N_S,C_i_p)];
    hold on
end
grid on
xlabel('$N_{\mathcal{S}}$','Interpreter','latex','FontSize',20);
ylabel('$^{||}C^{(I)}(N_\mathcal{S}; m)$','Interpreter','latex','FontSize',20);   
% title('Incremental','FontSize',15)

SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, plt, [], [], 8.8, 4, 1)
if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','parallel_incremental_complexity'),'-pdf')
    close(gcf);
end 

% Transfer ----------------------------------------------------------------
figure('Color',[1 1 1])
% hold on
p =  [];
for i = 1:numel(m)
    bar_sigma_m = exp(-alpha*N_S/(m(i)*N_K));
    factor = 1 - 0.5*(1 + N_K)*beta*(1 - bar_sigma_m);
    C_i_p = m(i)*N_K*c0*(1 - exp(-alpha*N_S/(m(i)*N_K)))/(1 - exp(-alpha));
    C_t_p = factor.*C_i_p;
    p = [p loglog(N_S,C_t_p)];
    hold on
end
grid on
xlabel('$N_{\mathcal{S}}$','Interpreter','latex','FontSize',20);
ylabel('$^{||}C^{(T)}(N_\mathcal{S}; m)$','Interpreter','latex','FontSize',20);   
% title('Transfer','FontSize',15)

SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, plt, [], [], 8.8, 4, 1)
if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','parallel_transfer_complexity'),'-pdf')
    close(gcf);
end 

% Collective --------------------------------------------------------------
figure('Color',[1 1 1])
% hold on
p = [];
for i = 1:numel(m)
    C_c = m(i)*c0*(1 - exp(-alpha*N_S))/(1 - exp(-alpha*m(i)));
    p = [p loglog(N_S,C_c)];
    hold on
end
legend(num2str(m(1)),num2str(m(2)))
grid on
xlabel('$N_{\mathcal{S}}$','Interpreter','latex','FontSize',20);
ylabel('$^{||}C^{(C)}(N_\mathcal{S}; m)$','Interpreter','latex','FontSize',20);   
% title('Collective','FontSize',15)

SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, plt, [], [], 8.8, 4, 1)
if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','parallel_collective_complexity'),'-pdf')
    close(gcf);
end 
%%
clearvars
clc

N_S   = 10^4;
N_K   = logspace(1,log10(N_S),log10(N_S));
alpha = 0.1;
beta  = 1./N_K;
m     = 1:1:100;
% m(mod(N_S,m) ~= 0) = NaN;
c0    = 100;

    

%%

P  = 100;
dt = 5*60;
e0 = P*dt;

% Incremental -------------------------------------------------------------
close all
figure
hold on
for i = 1:numel(N_K)
%     for j=1:numel(m)
%         C_i_p(j) = m(j)*(N_K(i)*c0*(1 - exp(-alpha*N_S./(m(j)*N_K(i))))./(1 - exp(-alpha)));
%     end
    C_i_p = m.*(N_K(i)*c0*(1 - exp(-alpha*N_S./(m.*N_K(i))))./(1 - exp(-alpha)));
    plot(m,C_i_p,'.')
end

xlabel('$m$','Interpreter','latex','FontSize',20);
ylabel('$^{||}C^{(I)}(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{4}} $','Interpreter','latex','FontSize',20);   
title('Incremental','FontSize',15)
grid minor

% Transfer ----------------------------------------------------------------
clc
fcolor = [0 0.4470 0.7410;
0.8500 0.3250 0.0980;
0.9290 0.6940 0.1250;
0.4940 0.1840 0.5560
0.4660 0.6740 0.1880; 
0.3010 0.7450 0.9330;
0.6350 0.0780 0.1840];

for i = 1:numel(N_K)
    for j=1:numel(m)
        bar_sigma_m(i,j) = exp(-alpha*N_S/(m(j)*N_K(i)));
        factor(i,j)      = 1 - 0.5*(1 + N_K(i))*beta(i)*(1 - bar_sigma_m(i,j));
        C_i_p(i,j)       = m(j)*N_K(i)*c0*(1 - exp(-alpha*N_S/(m(j)*N_K(i))))/(1 - exp(-alpha));
        C_t_p(i,j)       = factor(i,j).*C_i_p(i,j);
    end
end
C_c_p = m.*(c0*(1 - exp(-alpha*N_S))./(1 - exp(-alpha*m)));

close all
figure('Color',[1 1 1])
    subplot(2,1,1)
    semilogy(m,e0*C_i_p,'--','LineWidth',2);
    ax = gca;
    ax.ColorOrder = fcolor;
    hold on
    ax.ColorOrderIndex = 1;
    semilogy(m,e0*C_t_p,'-.','LineWidth',2);
    semilogy(m,e0*C_c_p,'k-','LineWidth',2)
    xlabel('$m$','Interpreter','latex','FontSize',20);
    ylabel('$E_{\mathcal{S}}(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{6}} $ [J]','Interpreter','latex','FontSize',20);   
    title('Total Energy VS. No. of Robots','FontSize',15)
    grid minor 
    xlim([min(m) max(m)])

aux = NaN(1,numel(N_K));
ax.ColorOrderIndex = 1;
p1 = plot(1,aux, 'LineWidth',3);
p2 = plot(1,aux(:,1),'k --','LineWidth',2);
p3 = plot(1,aux(:,1),'k -.','LineWidth',2);
p4 = plot(1,aux(:,1),'k -','LineWidth',2);
p = [p1; p2; p3; p4];
labels = cell(1,numel(N_K));
for i=1:numel(N_K)
    labels{i} = ['$N_{\mathcal{K}}=$' num2str(N_K(i))];
end
labels{i+1} = 'Incremental';
labels{i+2} = 'Transfer';
labels{i+3} = 'Collective';
legend(p,labels,'Interpreter','latex','FontSize',15)    
clear aux p labels

% subplot(2,1,2)
%     ax = gca;
%     ax.ColorOrder = fcolor;
%     semilogy(m,e0*C_i_p./m,'--','LineWidth',2);
%     hold on
%     ax.ColorOrderIndex = 1;
%     semilogy(m,e0*C_t_p./m,'-.','LineWidth',2);
%     semilogy(m,e0*C_c_p./m,'k-','LineWidth',2)
%     xlabel('$m$','Interpreter','latex','FontSize',20);
%     ylabel('$E_{\mathcal{S}}/m$ [J]','Interpreter','latex','FontSize',20);   
%     title('Energy per agent (-- Incremental | -. Transfer | - Collective) ','FontSize',15)
%     grid minor   
%     xlim([min(m) max(m)])
subplot(2,1,2)
    ax = gca;
    ax.ColorOrder = fcolor;
    semilogy(m,dt*C_i_p./m,'--','LineWidth',2);
    hold on
    ax.ColorOrderIndex = 1;
    semilogy(m,dt*C_t_p./m,'-.','LineWidth',2);
    semilogy(m,dt*C_c_p./m,'k-','LineWidth',2)
    xlabel('$m$','Interpreter','latex','FontSize',20);
    ylabel('$T_{\mathcal{S}}(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{6}}$ [s]','Interpreter','latex','FontSize',20);   
    title('Total Time VS. No. of Robots','FontSize',15)
    grid minor   
    xlim([min(m) max(m)])

aux = NaN(1,numel(N_K));
ax.ColorOrderIndex = 1;
p1 = plot(1,aux, 'LineWidth',3);
p2 = plot(1,aux(:,1),'k --','LineWidth',2);
p3 = plot(1,aux(:,1),'k -.','LineWidth',2);
p4 = plot(1,aux(:,1),'k -','LineWidth',2);
p = [p1; p2; p3; p4];
labels = cell(1,numel(N_K));
for i=1:numel(N_K)
    labels{i} = ['$N_{\mathcal{K}}=$' num2str(N_K(i))];
end
labels{i+1} = 'Incremental';
labels{i+2} = 'Transfer';
labels{i+3} = 'Collective';
legend(p,labels,'Interpreter','latex','FontSize',15)    
clear aux p labels   
%%



figure;plot(m,bar_sigma_m,'.')
legend("10","100","1000","10000")
xlabel('$m$','Interpreter','latex','FontSize',20);
ylabel('$\bar{\sigma}_m(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{4}}$','Interpreter','latex','FontSize',20);   
title('Knowledge','FontSize',15)
grid minor

%% Simulation *************************************************************
close all
% rng default; % For reproducibility

X = [randn(100,2)*rand(1,1) + randi(5)*ones(100,2);
     randn(100,2)*rand(1,1) - randi(5)*ones(100,2);
     randn(100,2)*rand(1,1) + randi(5)*ones(100,2)];
% plot(X,'k*')
% X = rand(100,2);
X = fcn_create_dummy_clusters();
plot(X(:,1),X(:,2),'k*')


%%
N_S   = 100;
N_K   = 5;
alpha = 0.1;
close all
% X = rand(N_S,2);
clusters = N_K;
opts = statset('Display','final');
% [idx,C] = kmeans(X,clusters,'Distance','cityblock',...
%     'Replicates',5,'Options',opts);
[idx,C] = kmeans(X,clusters);

figure;
% plot(X,'ko')
% hold on
for k=1:clusters
plot(X(idx==k,1),X(idx==k,2),'.','MarkerSize',12)
hold on
end
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
% legend('Cluster 1','Cluster 2','Centroids',...
%        'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%%
close all
clc
N_S   = 200;
N_K   = 5;
rng default; % For reproducibility
X = rand(N_S,2);
% X = fcn_create_dummy_clusters();
% idx = dbscan(X,0.069,2);
% idx = spectralcluster(X,5);
idx = kmedoids(X,N_K);
% figure
% for k=1:N_K
%     plot(X(idx==k,1),X(idx==k,2),'*','MarkerSize',12)
%     hold on
% end

for k=1:1%N_K
    % d = pdist([X(idx==k,1),X(idx==k,2)]);
    % d = d/max(d);
    D       = squareform(pdist([X(idx==k,1),X(idx==k,2)])); % distance between pairs of tasks
%     D       = D./max(D,[],'all');   % normalized distance
    % alpha       = D./sum(D,'all');   % normalized distance
%     alpha = normr(D);

    figure('Color',[1 1 1])
    C         = 0;
    cj        = zeros(size(D,1),1);
    sigma_bar = zeros(size(D,1),1);
    for j = 1:size(D,1)
        %alpha        = D(j,:)./sum(D(j,1:(j-1))); 
        %sigma_bar(j) = exp(-sum(alpha(1:(j-1))));
        if j==1
            sigma_bar(j) = 1;
            SumAlpha(j) = 0;
        else
            SumAlpha(j)     = sum(D(j,1:(j-1)));%/max(D(j,1:(j-1)));
            sigma_bar(j) = exp(-SumAlpha(j));
        end
        disp(SumAlpha(j))
        cj(j)        = c0*sigma_bar(j);
        n            = 0:0.1:cj(j);
        p = plot3(n, j*ones(numel(n),1), sigma_bar(j)*exp(-6/cj(j)*n),'LineWidth',2);
        hold on
        p = fill3(n([1 1:end end]), j*ones(numel(n)+2,1), [0 sigma_bar(j)*exp(-6/cj(j)*n) 0],p.Color, 'FaceAlpha', 0.1);
        C = C + cj(j);  
    end
    plot3(zeros(size(D,1),1),1:size(D,1),sigma_bar ,'r-','LineWidth', 2,'color',[0.2 0.2 0.2])
    plot3(cj,1:size(D,1),zeros(size(D,1),1),'b-','LineWidth', 2,'color',[0.2 0.2 0.2])
    xlabel('$c_j^{(I)}$','Interpreter','latex','FontSize',20);
    ylabel('$N_{j}$','Interpreter','latex','FontSize',20);   
    zlabel('$\bar{\sigma}_{j}^{(I)}$','Interpreter','latex','FontSize',20);   
    ylim([1 size(D,1)])
    title(['Cluster k = ' num2str(k)])
end
% figure
% plot(X(:,1),X(:,2),'k.')
%%
clc
close all

d = 2;
k = 5;
n = 1000;
% Generate data 
[X,label] = kmeansRnd(d,k,n);
X = X';
plot(X(:,1),X(:,2),'k*')