clc
N_K = 10;
N_S = 1:1:10000;
alpha = 0.1;
beta = 1/N_K;
m = 10:10:100;
c0 = 100;
% Incremental -------------------------------------------------------------
close all
figure('Color',[1 1 1])
hold on
for i = 1:numel(m)
    C_i_p = m(i)*N_K*c0*(1 - exp(-alpha*N_S/(i*N_K)))/(1 - exp(-alpha));
    plot(N_S,C_i_p)
end

xlabel('$N_{\mathcal{S}}$','Interpreter','latex','FontSize',20);
ylabel('$^{||}C^{(I)}(N_\mathcal{S}; m)$','Interpreter','latex','FontSize',20);   
title('Incremental','FontSize',15)
grid on
% Transfer ----------------------------------------------------------------
figure('Color',[1 1 1])
hold on
for i = 1:numel(m)
    bar_sigma_m = exp(-alpha*N_S/(m(i)*N_K));
    factor = 1 - 0.5*(1 + N_K)*beta*(1 - bar_sigma_m);
    C_i_p = m(i)*N_K*c0*(1 - exp(-alpha*N_S/(i*N_K)))/(1 - exp(-alpha));
    C_t_p = factor.*C_i_p;
    plot(N_S,C_t_p)
end
xlabel('$N_{\mathcal{S}}$','Interpreter','latex','FontSize',20);
ylabel('$^{||}C^{(T)}(N_\mathcal{S}; m)$','Interpreter','latex','FontSize',20);   
title('Transfer','FontSize',15)
grid on

%% ************************************************************************
%    Total time and energy consumption in parallel learning paradigms     *
% *************************************************************************
clearvars
clc

N_S   = 1E4;
N_K   = logspace(1,log10(N_S),log10(N_S));
alpha = 0.1;
beta  = 1./N_K;
m     = 1:1:100;
% m(mod(N_S,m) ~= 0) = NaN;
c0    = 100;
P     = 100;
dt    = 5*60;
e0    = P*dt;

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
%     subplot(2,1,1)
    e0 = 1;
    p1 = semilogy(m,e0*C_i_p,'--','LineWidth',2);
    ax = gca;
    ax.ColorOrder = fcolor;
    hold on
    ax.ColorOrderIndex = 1;
    p2 = semilogy(m,e0*C_t_p,'-.','LineWidth',2);
    p3 = semilogy(m,e0*C_c_p,'k-','LineWidth',2);
    
    p = [p1; p2; p3];
    xlabel('$m$','Interpreter','latex','FontSize',20);
    ylabel('$E_{\mathcal{S}}(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{6}} $ [J]','Interpreter','latex','FontSize',20);   
%     title('Total Energy VS. No. of Robots','FontSize',15)
    grid minor 
    xlim([min(m) max(m)])

aux = NaN(1,numel(N_K));
ax.ColorOrderIndex = 1;
h1 = plot(1,aux, 'LineWidth',3);
h2 = plot(1,aux(:,1),'k --','LineWidth',2);
h3 = plot(1,aux(:,1),'k -.','LineWidth',2);
h4 = plot(1,aux(:,1),'k -','LineWidth',2);
h = [h1; h2; h3; h4];
labels = cell(1,numel(N_K));
for i=1:numel(N_K)
    labels{i} = ['$N_{\mathcal{K}}=$' num2str(N_K(i))];
end
labels{i+1} = 'Incremental';
labels{i+2} = 'Transfer';
labels{i+3} = 'Collective';
leg = legend(h,labels,'Interpreter','latex','FontSize',15);  
clear aux  labels

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

SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, plt, leg, [], 8.8, 4, 1) %(fig, ax, plt, leg, tx, text_width, k_scaling, k_width_height)

if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','parallel_total_energy'),'-pdf')
    close(gcf);
end    

%%
figure
semilogy(m,C_t_p(1,:)./C_c_p,'k-','LineWidth',2);
hold on
semilogy(m,C_c_p,'r-','LineWidth',2);
%% ************************************************************************
%    Total time and energy consumption in parallel learning paradigms     *
% *************************************************************************
clearvars
clc

N_S   = 1E4;
N_K   = 10;
alpha = 0.1;
beta  = 1./N_K;
m     = 1:1:100;
% m(mod(N_S,m) ~= 0) = NaN;
c0    = 100;
P     = 100;
dt    = 5*60;
e0    = P*dt;

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
%         C_t_p(i,j)       = factor(i,j).*C_i_p(i,j);
        C_t_p(i,j)       = -1/(N_S*m(i))*log(factor(i,j).*C_i_p(i,j)/(N_S*c0));
    end
end
C_c_p = m.*(c0*(1 - exp(-alpha*N_S))./(1 - exp(-alpha*m)));

close all
figure('Color',[1 1 1])
%     subplot(2,1,1)
    p1 = semilogy(m,e0*C_i_p,'k--','LineWidth',2);
    ax = gca;
    ax.ColorOrder = fcolor;
    hold on
    ax.ColorOrderIndex = 1;
    p2 = semilogy(m,e0*C_t_p,'k-.','LineWidth',2);
    p3 = semilogy(m,e0*C_c_p,'k-','LineWidth',2);
    p = [p1; p2; p3];
    xlabel('$m$','Interpreter','latex','FontSize',20);
    ylabel('$E $ [J]','Interpreter','latex','FontSize',20);   
%     title('Total Energy VS. No. of Robots','FontSize',15)
    grid minor 
    xlim([min(m) max(m)])

aux = NaN(1,numel(N_K));
ax.ColorOrderIndex = 1;
h1 = plot(1,aux, 'k', 'LineWidth',3);
h2 = plot(1,aux(:,1),'k --','LineWidth',2);
h3 = plot(1,aux(:,1),'k -.','LineWidth',2);
h4 = plot(1,aux(:,1),'k -','LineWidth',2);
h = [h2; h3; h4];
% labels = cell(1,numel(N_K));
% for i=1:numel(N_K)
%     labels{i} = ['$N_{\mathcal{K}}=$' num2str(N_K(i))];
% end
i = 0;
labels{i+1} = 'Incremental';
labels{i+2} = '$\gamma$';
labels{i+3} = 'Collective';
leg = legend(h,labels,'Interpreter','latex','FontSize',15);  
clear aux  labels

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

SAVE_FIG = 0;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, plt, leg, [], 8.8, 4, 1) %(fig, ax, plt, leg, tx, text_width, k_scaling, k_width_height)

if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','consortium'),'-pdf')
    close(gcf);
end    

%%

figure('Color',[1 1 1])
    ax = gca;
    ax.ColorOrder = fcolor;
    p1 = semilogy(m,dt*C_i_p./m,'--','LineWidth',2);
    hold on
    ax.ColorOrderIndex = 1;
    p2 = semilogy(m,dt*C_t_p./m,'-.','LineWidth',2);
    p3 = semilogy(m,dt*C_c_p./m,'k-','LineWidth',2);
    p = [p1; p2; p3];
    xlabel('$m$','Interpreter','latex','FontSize',20);
    ylabel('$T_{\mathcal{S}}(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{6}}$ [s]','Interpreter','latex','FontSize',20);   
%     title('Total Time VS. No. of Robots','FontSize',15)
    grid minor   
    xlim([min(m) max(m)])

aux = NaN(1,numel(N_K));
ax.ColorOrderIndex = 1;
h1 = plot(1,aux, 'LineWidth',3);
h2 = plot(1,aux(:,1),'k --','LineWidth',2);
h3 = plot(1,aux(:,1),'k -.','LineWidth',2);
h4 = plot(1,aux(:,1),'k -','LineWidth',2);
h = [h1; h2; h3; h4];
labels = cell(1,numel(N_K));
for i=1:numel(N_K)
    labels{i} = ['$N_{\mathcal{K}}=$' num2str(N_K(i))];
end
labels{i+1} = 'Incremental';
labels{i+2} = 'Transfer';
labels{i+3} = 'Collective';
leg = legend(h,labels,'Interpreter','latex','FontSize',15);
clear aux labels   

SAVE_FIG = 1;
fig = gcf;
ax  = gca;
plt = p;
fcn_scrpt_prepare_graph(fig, ax, plt, leg, [], 8.8, 4, 1) %(fig, ax, plt, leg, tx, text_width, k_scaling, k_width_height)

if SAVE_FIG == 1
    export_fig(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','parallel_total_time'),'-pdf')
    close(gcf);
end 
%%



figure;plot(m,bar_sigma_m,'.')
legend("10","100","1000","10000")
xlabel('$m$','Interpreter','latex','FontSize',20);
ylabel('$\bar{\sigma}_m(m; N_\mathcal{K}) |_{N_\mathcal{S}=10^{4}}$','Interpreter','latex','FontSize',20);   
title('Knowledge','FontSize',15)
grid minor