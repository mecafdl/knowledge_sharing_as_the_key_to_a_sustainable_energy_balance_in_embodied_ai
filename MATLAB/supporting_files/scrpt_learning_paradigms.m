%% - Functions folder in the same folder
cd(fileparts(matlab.desktop.editor.getActiveFilename)); % use this to CD
addpath(genpath(pwd))

% Threshold to consider a skill learned
epsilon = 0.01;

% Fundamental complexity (AKA max number of episodes to learn any skill)
c0 = 100;
% Trial episodes
n  = 0:0.01:c0; 

% Total number of skills
N_S = 8*64;

% Number of clusters
N_K = 4;

% Number of skills per cluster
N_Z = N_S/N_K;

% Basic learning rate
alpha = -log(epsilon)/c0;%0.05;

delta = -log(epsilon)/N_Z;%0.05;

eta   = 0.1;

f = @(eta, N_zeta) eta*N_zeta+1;
g = @(delta, N_zeta) exp(-delta*N_zeta);

%% number of robots available
m = 32;
% skill_labels = cell(N_Z/m,1);
% for i=1:N_Z/m
%     skill_labels{i,1} = ['s_{',num2str(i),'}'];
% end

%% **************************************************************************
% Dynamics of ISOLATED LEARNING 
% *************************************************************************
clc
close all

clear c_jk_iso_episodes
entry = 1;
% Loop over robots
for m = [2,4,8,16,32,64,128]
    fig       = figure('color','w');
    % Transfer learning factor
    beta_k   = 0;
    c_jk_iso = zeros(N_Z, N_K);
    % Loop over clusters
    for k = 1:N_K
        disp(['Cluster ',num2str(k),' -------------------------'])
        subplot(1,N_K,k)
        % Loop over skills
        for N_zeta = 0:(N_Z/m)-1
            j        = N_zeta+1;
            bsigma_0 = ones(m,1);
            a        = -alpha;%*((1- beta_k)^(-1))*f(eta, N_zeta);
            F        = a*eye(m);
            bsigma   = ode4(@(n,bsigma) F*bsigma, n, bsigma_0);
            semilogy(n,bsigma,'LineWidth',2)
            hold on
            c_jk_iso(j, k) = -log(epsilon)/alpha;
        end
        
        p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
        plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
        title(['$(r_i,\mathcal{Z}_{',num2str(k),'})$'],'FontSize',25,'Interpreter','latex')
        xlabel('n','FontSize',25)
        ylabel(['$\bar{\boldmath{\sigma}}^{(IsL)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
        xlim([0 n(end)])
        ylim([1E-3 1E0])
        xticks([0 50 100])
        xticklabels({'0','', '$c_0$'})
        yticks([1E-3 1E-2 1E-1 1E0])
        yticklabels({'','$\epsilon$', '', '1'})
        set(gca,'TickLabelInterpreter','latex')
    end
    % legend(skill_labels)
    for ax=1:N_K
        fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
    end
    fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
    pause(1)
    tightfig(fig);
    c_jk_iso_episodes(entry,:) = [sum(c_jk_iso,1), sum(c_jk_iso,'all')] ;
    entry      = entry+1; 
end
%%

fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_isolated_learning'));
disp('Printing done!')
%% **************************************************************************
% Dynamics of INCREMENTAL LEARNING 
% *************************************************************************
clc
close all

clear c_jk_il_episodes
entry = 1;
% Loop over robots
for m = [2,4,8,16,32,64,128]
    fig     = figure('color','w');
    % Transfer learning factor
    beta_k  = 0;
    c_jk_il = zeros(N_Z,N_K);
    % Loop over clusters
    for k = 1:N_K
        disp(['Cluster ',num2str(k),' -------------------------'])
        subplot(1,N_K,k)
        % Loop over skills
        for N_zeta = 0:(N_Z/m)-1
            j        = N_zeta+1;
            bsigma_0 = (1- beta_k)*g(delta, N_zeta).*ones(m,1);
            a        = -alpha*((1- beta_k)^(-1))*f(eta, N_zeta);
            F        = a*eye(m);
            bsigma   = ode4(@(n,bsigma) F*bsigma, n, bsigma_0);
            semilogy(n,bsigma,'LineWidth',2)
            hold on
            c_jk_il(j, k) = -(log(epsilon) + delta*N_zeta)/(alpha*f(eta, N_zeta));
        end
        disp(['Skills seen:' num2str(j)]);
        p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
        plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
        title(['$(r_i,\mathcal{Z}_{',num2str(k),'})$'],'FontSize',25,'Interpreter','latex')
        xlabel('n','FontSize',25)
        ylabel(['$\bar{\boldmath{\sigma}}^{(IL)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
        xlim([0 n(end)])
        ylim([1E-3 1E0])
        xticks([0 50 100])
        xticklabels({'0','', '$c_0$'})
        yticks([1E-3 1E-2 1E-1 1E0])
        yticklabels({'','$\epsilon$', '', '1'})
        set(gca,'TickLabelInterpreter','latex')
    end
    % legend(skill_labels)
    for ax=1:N_K
        fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
    end
    fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
    pause(1)
    tightfig(fig);
    c_jk_il_episodes(entry,:) = [sum(c_jk_il,1), sum(c_jk_il,'all')];
    entry = entry + 1;
end
%%

fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_incremental_learning'));
disp('Printing done!')
%% **************************************************************************
% Dynamics of TRANSFER + INCREMENTAL LEARNING
% *************************************************************************
clc
close all
warning('CHECK THE INFLUENCE OF ROBOTS IN PARALLEL ON BETA')
clear c_jk_til_episodes
entry = 1;
% Loop over robots
for m = [2,4,8,16,32,64,128]
    fig       = figure('color','w');
    c_jk_itl  = zeros(N_Z, N_K);
    % Loop over clusters
    for k = 1:N_K
        disp(['Cluster ',num2str(k),' -------------------------'])
        subplot(1,N_K,k)
        % Transfer learning factor
        %beta_k = (k-1)/N_K;
        beta_k = (1/m)*(k-1)/N_K;
        %beta_k   = (N_zeta)/N_S;
        % Loop over skills
        for N_zeta = 0:(N_Z/m)-1
            j        = N_zeta+1;
            bsigma_0 = (1- beta_k)*g(delta, N_zeta).*ones(m,1);
            a        = -alpha*((1- beta_k)^(-1))*f(eta, N_zeta);
            F        = a*eye(m);
            bsigma   = ode4(@(n,bsigma) F*bsigma, n, bsigma_0);    
            semilogy(n,bsigma,'LineWidth',2)
            ylim([-0.1 1.1])
            hold on
            c_jk_itl(j, k) = -(log(epsilon*(1- beta_k)^(-1)) + delta*N_zeta)/(alpha*(1- beta_k)^(-1)*f(eta, N_zeta));
        end
        p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
        plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
        title(['$(r_i,\mathcal{Z}_{',num2str(k),'})$'],'FontSize',25,'Interpreter','latex')
        xlabel('n','FontSize',25)
        ylabel(['$\bar{\boldmath{\sigma}}^{(TIL)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
        xlim([0 n(end)])
        ylim([1E-3 1E0])
        xticks([0 50 100])
        xticklabels({'0','', '$c_0$'})
        yticks([1E-3 1E-2 1E-1 1E0])
        yticklabels({'','$\epsilon$', '', '1'})
        set(gca,'TickLabelInterpreter','latex')
    end
    for ax=1:4
        fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
    end
    fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
    pause(1)
    tightfig(fig);
    c_jk_til_episodes(entry,:) = [sum(c_jk_itl,1), sum(c_jk_itl,'all')] ;
    entry                      = entry + 1; 
end
%%
fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_incremental_transfer_learning'));
disp('Printing done!')
%% **************************************************************************
% Dynamics of COLLECTIVE LEARNING
% *************************************************************************
% * NOTE: all m robots placed in one cluster at a time concurrently 
%         exchanging knowledge
clc
close all

clear c_jk_cl_episodes
entry = 1;
% Loop over robots
for m = [2,4,8,16,32,64,128]   
    A_full  = ones(m) - eye(m);
    A       = A_full;
    fig     = figure('color','w');
    % Coupling
    % gamma = 1*0.005;
    gamma   = -1*0.01;
    r       = double(gamma==0) + double(gamma~=0)*m;
    c_jk_cl = zeros(N_Z, N_K);
    % Loop over clusters
    for k = 1:N_K
        disp(['Cluster ',num2str(k),' -------------------------'])
        subplot(1,N_K,k)
        beta_k = (k-1)/N_K;
        % Loop over skills
        for N_zeta = 0:N_Z/m-1    
            j        = N_zeta+1;
            bsigma_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
            a        = -alpha*((1- beta_k)^(-1))*f(eta, r*N_zeta);
            F        = a*eye(m) + gamma*A;
            lambda_F = eig(F);
            disp(lambda_F)
            if any(real(lambda_F)>0)
                warning('Unstable!!!')
                return
            end
            bsigma   = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
            c_jk_cl(j, k) = n(min(find(bsigma(1,:)<epsilon)));
            semilogy(n,bsigma,'LineWidth',3)
            xlim([0 n(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
            hold on
        end
        p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
        plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
        title(['$(r_i,\mathcal{Z}_{',num2str(k),'}$)'],'FontSize',25,'Interpreter','latex')
        xlabel('n','FontSize',25)
        ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    end
    for ax=1:N_K
        fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
    end
    fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
    pause(1)
    tightfig(fig);
    c_jk_cl_episodes(entry,:) = [sum(c_jk_cl,1), sum(c_jk_cl,'all')] ;
    entry                     = entry + 1; 
end
%% Alternative robot distribution: EQUAL robots per cluster
close all
clear c_jk_cl_dist_episodes
entry = 1;
% Loop over robots
for m = [4,8,16,32,64,128]
    A_full = ones(m) - eye(m);
    A      = A_full;
    % Inter cluster scaling matrix
    %beta_k = 1/N_K;
    beta_k = 1/(m*N_K);
    B      = beta_k*A;
    
    % Robots concurrently exchanging knowledge
    r = m;
    
    clc

    beta_k = (1/N_K);
    B      = beta_k*A;
    for i=1:N_K
       B((m/N_K)*(i-1) + 1:(m/N_K)*i,(m/N_K)*(i-1) + 1:(m/N_K)*i) = ones(m/N_K) - eye(m/N_K);
    end
    B = triu(B) + transpose(triu(B));
    
    fig = figure('color','w');
    c_jk_cl_dist  = zeros(N_S/m, 1);%zeros(N_Z, N_K);
    gamma    = -0.01;%*(-(a)/max(eig(abs(A))));
    r        = double(gamma==0) + double(gamma~=0)*m;
    
%     for k = 1:(N_S/m)  
%         disp(['Batch ',num2str(k),' -------------------------'])
%     %     subplot(2,N_K,k)        
%         N_zeta   = k-1;
%         bsigma_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
%         a        = -alpha*f(eta, r*N_zeta);%*(1- beta_k)^(-1);
%         F        = a*eye(m) + gamma*A.*B;
%         lambda_F = eig(F);
%         bsigma   = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
%         c_jk_cl_dist(k) = n(min(find(bsigma(1,:)<epsilon)));
%         disp(lambda_F)        
%         semilogy(n,bsigma,'LineWidth',2)
%         xlim([0 n(end)])
%         ylim([1E-3 1E0])
%         xticks([0 50 100])
%         xticklabels({'0','', '$c_0$'})
%         yticks([1E-50 1E-2 1E-1 1E0])
%         yticklabels({'','$\epsilon$', '', '1'})
%         set(gca,'TickLabelInterpreter','latex')
%         hold on
%     end
    for N_zeta = 0:(N_S/m)-1
        % Beta adapts at every cycle
        % ============================================
        beta_k = r*N_zeta/N_S
%         beta_k = (r*N_zeta/N_Z)*(1/N_K)%min(r*N_zeta/N_Z,1)*(1/N_K)
        B      = beta_k*A;
        for i=1:N_K
           B((m/N_K)*(i-1) + 1:(m/N_K)*i,(m/N_K)*(i-1) + 1:(m/N_K)*i) = ones(m/N_K) - eye(m/N_K);
        end
        B = triu(B) + transpose(triu(B));
        % ============================================


        j        = N_zeta + 1;
        disp(['Batch ',num2str(j),' -------------------------'])
        bsigma_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
        a        = -alpha*f(eta, r*N_zeta);%*(1- beta_k)^(-1);
        F        = a*eye(m) + gamma*A.*B;
        lambda_F = eig(F);
        bsigma   = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
        c_jk_cl_dist(j) = n(min(find(bsigma(1,:)<epsilon)));
        disp(lambda_F)        
        semilogy(n,bsigma,'LineWidth',2)
        xlim([0 n(end)])
        ylim([1E-3 1E0])
        xticks([0 50 100])
        xticklabels({'0','', '$c_0$'})
        yticks([1E-50 1E-2 1E-1 1E0])
        yticklabels({'','$\epsilon$', '', '1'})
        set(gca,'TickLabelInterpreter','latex')
        hold on
    end

    p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
    % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,k}$'],'FontSize',25,'Interpreter','latex')
    fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 18/2, 3, 1)
    pause(1)
    tightfig(fig);
    c_jk_cl_dist_episodes(entry) = sum(c_jk_cl_dist);
    entry                        = entry + 1; 
end
% % Inter cluster scaling matrix
% beta_k = 1/N_K;
% B      = beta_k*A;
% 
% % Robots concurrently exchanging knowledge
% r = m;
% 
% clc
% for i=1:N_K
%    row = (2*(i-1) + 1);
%    B(row, row+1) = 1;
% end
% B = triu(B) + transpose(triu(B));
% 
% figure('color','w')
% c_jk_cl  = zeros(N_Z, N_K);
% for k = 1:(N_S/m)  
%     disp(['Batch ',num2str(k),' -------------------------'])
%     subplot(2,N_K,k)        
%     N_zeta   = k-1;
%     bsigma_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
%     a        = -alpha*f(eta, r*N_zeta);%*(1- beta_k)^(-1);
%     gamma    = -0.01;%*(-(a)/max(eig(abs(A))));
%     F        = a*eye(r) + gamma*A.*B
%     lambda_F = eig(F);
%     bsigma   = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
%     disp(lambda_F)        
%     plot(n,bsigma,'LineWidth',2)
%     hold on
%     p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
%     plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
%     title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
%     xlabel('n','FontSize',25)
%     ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
%     xlim([0 n(end)])
%     ylim([-0.1 1.1])    
% end
%% Total number of episodes
close all

factor = [2,4,8,16,32,64,128]'*105E3;
fig = figure('color','w');
p0 = semilogy([2,4,8,16,32,64,128],c_jk_iso_episodes(:,end),'o-');
hold on
p1 = plot([2, 4, 8, 16, 32, 64,128],c_jk_il_episodes(:,end),'o-');
p2 = plot([2, 4, 8, 16, 32, 64,128],c_jk_til_episodes(:,end),'o-');
p3 = plot([2, 4, 8, 16, 32, 64,128],c_jk_cl_episodes(:,end),'o-');
p4 = plot(   [4, 8, 16, 32, 64,128],c_jk_cl_dist_episodes,'o-');
plot(32*ones(size(1E1:100:1E5)),1E1:100:1E5,'k--','LineWidth',3)
xticks([2,4,8,16,32,64,128])
xlabel('m','FontSize',25)
ylabel('$C_\mathcal{S}$','FontSize',25,'Interpreter','latex')
% leg = legend('IsL','IL','TIL','CL','CL2');
leg = legend('Isolated','Incremental','Transfer + Incremental','Collective','Collective (distributed)');
fcn_scrpt_prepare_graph_science_std(fig, gca, [p0, p1, p2, p3, p4], leg, [], 18/2, 3, 1)
leg.Location = 'northeast';
leg.Box = 'on';
fig = gcf;           % generate a figure
tightfig(fig);
pause(1)
%%
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','total_energy_per_n_robots'));
disp('Printing done!')

%% Total energetic cost
close all

factor = [2,4,8,16,32,64,128]'*105E3;
fig = figure('color','w');
p0 = semilogy([2,4,8,16,32,64,128],factor.*c_jk_iso_episodes(:,end),'o-');
hold on
p1 = plot([2, 4, 8, 16, 32, 64,128],factor.*c_jk_il_episodes(:,end),'o-');
p2 = plot([2, 4, 8, 16, 32, 64,128],factor.*c_jk_til_episodes(:,end),'o-');
p3 = plot([2, 4, 8, 16, 32, 64,128],factor.*c_jk_cl_episodes(:,end),'o-');
p4 = plot(   [4, 8, 16, 32, 64,128],factor(2:end).*c_jk_cl_dist_episodes','o-');
% plot(32*ones(size(1E1:100:1E5)),1E1:100:1E5,'k--','LineWidth',3)
% xticks([2,4,8,16,32,64,128])
xlim([1 128])
xlabel('m','FontSize',25)
ylabel('$E_\mathcal{S}~[J]$','FontSize',25,'Interpreter','latex')
% leg = legend('IsL','IL','TIL','CL','CL2');
leg = legend('Isolated','Incremental','Transfer + Incremental','Collective','Collective (distributed)');
fcn_scrpt_prepare_graph_science_std(fig, gca, [p0, p1, p2, p3, p4], leg, [], 18/2, 3, 1)
leg.Location = 'northeast';
leg.Box = 'on';
fig = gcf;           % generate a figure
tightfig(fig);
pause(1)
%%
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','total_episodes_per_n_robots'));
disp('Printing done!')





%% Alternative robot distribution for collective learning

% Robots concurrently exchanging knowledge in one cluster and moving to the 


clc
close all
fig = figure('color','w');


r        = m;
c_jk_cl  = zeros(N_Z/m, N_K);
kappa = zeros(1,N_Z+1);
for k = 1:N_K
    disp(['Cluster ',num2str(k),' -------------------------'])
    subplot(1,N_K,k)
    beta_k = (k-1)/N_K;
    for N_zeta = 0:N_Z/m-1    
        j        = N_zeta+1;
        bsigma_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
        a        = -alpha*f(eta, r*N_zeta)*(1- beta_k)^(-1);
        gamma    = 0.005;%-0.3*(-(a)/max(eig(abs(A))));
        F        = a*eye(r) + gamma*A;
        lambda_F = eig(F)
        bsigma   = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
%         for i=0:m-1
%             c_jk_cl(j+i, k) = n(min(find(bsigma(i+1,:)<epsilon))); 
%         end
%         for r_i=1:r
%             c_jk_cl(r_i, k) = n(min(find(bsigma(r_i,:)<epsilon)));
%         end
        c_jk_cl(j, k) = n(min(find(bsigma(1,:)<epsilon)));
        semilogx(n,bsigma,'LineWidth',3)
        hold on
%         kappa(N_zeta + 1)  = sum(bsigma);
    end
    p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 n(end)])
    ylim([-0.1 1.1])
end
for ax=1:4
    fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
end
fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
pause(1)
tightfig(fig)
%%
fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_collective_learning'));
disp('Printing done!')
%%
for k = 1:N_K
    A = G*A_i;
    disp('A:');
    disp(A)
    disp('Eigenvalues:')
    disp([(eig(A)),(eig(A_i)),abs(eig(A))-abs(eig(A_i))]) 
    
    
    disp('Det:')    
    disp(det(A))
%     close all
    learning_type_1 = 'collective_sat';
    % (t, x, A, alpha, s, c0, rate, threshold, type)
    if cycle ==1
        var = 1;
      
    else
        var = (cycle-1)*m;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha, s, c0, tau, epsilon, learning_type_1), ...
                                             n, ...
                                             1-(cycle-1)/(N/m)*ones(m,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(n,x,'LineWidth',2);
    hold on
end
plot(n,exp(A_i(1,1)*n),'k--','LineWidth',2)
p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 n(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')


%% **************************************************************************
% Dynamics of colletive learning 
% *************************************************************************
clc
close all
clear A

% This variable determines when knowlege collection is completed
epsilon = 0.01;

% Number of skills
N = 10;

% Number of robots
m = 2;

% Skill index
s         = transpose(1:N);

% Skill vector
Nj        = 0:N;

% Rate at which knowledge is depleted per skill acquired
% *NOTE: determined to satisfy <threshold = exp(-alpha*N)>
alpha     = -log(epsilon)/N;  % log(0.01)/N;%(rate/N);

% Initial values of the subsequent skills based on alpha
bsigma_0  = exp(-alpha*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:c0*N;
n  = 0:0.01:100; 

% Initial complexity
c0        = 100;

% Initial knowledge aquisition rate per episode
% *NOTE: determined to satisfy <threshold = exp(-(rate/c0)*c0)>
tau       = -log(epsilon)*ones(m,1);
A_i       = -diag(tau/c0);

close all
%theta          = 1:15;%&10*rand(N*(N+1)/2,1);
% theta          = normrnd(5*ones(N*(N+1)/2,1),0.33*ones(N*(N+1)/2,1));
theta          = 1*ones(m*(m+1)/2,1);
% theta(N+1:end) = rand(N*(N+1)/2 - N,1);
gamma           = 1;%rand(1);
theta(m+1:end) = gamma*ones(m*(m+1)/2 - m,1);
ind            = find(triu(ones(m,m),1));
L              = zeros(m) + diag(theta(1:m));
L(ind)         = theta(m+1:end);
G              = L + triu(L,1)';
eig(G);
% G = (L')*L


X = NaN(m,numel(n),N/m);
fig = figure('Color','w');
fig.WindowState = 'maximized';
for cycle = 1:N/m
    A = G*A_i;
    disp('A:');
    disp(A)
    disp('Eigenvalues:')
    disp([(eig(A)),(eig(A_i)),abs(eig(A))-abs(eig(A_i))]) 
    
    
    disp('Det:')    
    disp(det(A))
%     close all
    learning_type_1 = 'collective_sat';
    % (t, x, A, alpha, s, c0, rate, threshold, type)
    if cycle ==1
        var = 1;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha, s, c0, tau, epsilon, learning_type_1), ...
                                             n, ...
                                             ones(m,1));        
    else
        var = (cycle-1)*m;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha, s, c0, tau, epsilon, learning_type_1), ...
                                             n, ...
                                             1-(cycle-1)/(N/m)*ones(m,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(n,x,'LineWidth',2);
    hold on
end
plot(n,exp(A_i(1,1)*n),'k--','LineWidth',2)
p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 n(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')

%%
% -------------------------------------------------------------------------
learning_type_2 = 'collective';
% (t, x, A, alpha, s, c0, rate, threshold, type)
x_nonebounded = ode4(@(t,x) ...
      fcn_general_dynamics(t, x, A, alpha, s, c0, tau, epsilon, learning_type_2), ...
                                     n, ...
...
ones(N,1));

figure('Color','w')
p = plot(n,x_nonebounded,':','LineWidth',1.5);
hold on
set(gca,'ColorOrderIndex',1)
p = plot(n,x,'LineWidth',2);
plot(n,exp(A_i(1,1)*n),'k--','LineWidth',2)
p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 n(end)])
ylim([-0.1 1.1])
H = gcf;% or H=gca
H.WindowState = 'maximized';

% figure('Color','w')
%     p   = bar3(abs(A));
%     colorbar off
%     colormap((jet))
%     for k = 1:length(p)
%         zdata = p(k).ZData;
%         p(k).CData = zdata;
%         p(k).FaceColor = 'interp';
%     end
%     xlabel('X')
%     ylabel('Y')
%     
%     xlim([1 size(A,1)])
%     ylim([1 size(A,2)])   
%%
 
close all
s         = [1:N]';

Nj        = 0:N;
alpha     = -log(epsilon)/N;%-log(0.01)/N;%(rate/N);
% alpha     = (rate/N);
bsigma_0  = exp(-alpha*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:c0*N;
n  = 0:0.01:1000; 
learning_type = 'incremental_3';
x = ode4(@(t,x) fcn_general_dynamics(t, x, A, alpha, s, c0, tau, epsilon,...
                learning_type), ...
                                     n, ...
                                     zeros(N,1) +1*bsigma_0(1:end-1)');
x = x';
figure('Color','w')
if strcmp(learning_type,'incremental_2')
%     p = plot(n,bsigma_0(1:end-1)'-x,'LineWidth',2);
    p = plot(n,x,'LineWidth',2);
else
    p = plot(n, x,'LineWidth',2);
end
hold on

% p = plot(n,(1-bsigma_0(1:end-1))' - x,'LineWidth',2);
% p = plot(n, - x,'LineWidth',2);
% hold on
plot(n(end),bsigma_0(1:end-1),'*')
xlabel('Episodes [n]','FontSize',25)
ylabel('Knowledge ','FontSize',25)
% xlim([0 sum(c0*bsigma_0)])
ylim([0 1])
%
title('Incremental')

% clear indx
% for j = 1:size(x,1)
%     delta  = x(j,:) - bsigma_0(j);
%     indx(j) = find(abs(delta) < threshold,1,'first');
% %     disp(n(indx(j)))
% end
% figure('Color','w')
% plot(1:N,c0*bsigma_0(1:end-1),'--','LineWidth',2)
% hold on
% plot(1:N,[n(indx(1)), diff(n(indx(:)))],'LineWidth',2)
% xlabel('Skills','FontSize',25)
% ylabel('Complexity ','FontSize',25)
% legend('Ideal','Actual','FontSize',10)
% xlim([1 N])


%%

sig_bar = @(Nj,alpha) exp(-alpha*Nj);
episodes = 0:c0;
clear indx
clc
close all
figure('Color','w')
% plot(Nj,sig_bar(Nj,(5/N)),'LineWidth',3)
% hold on
semilogy(episodes,0.01*ones(size(episodes)),'k:','LineWidth',3) % Threshold for skill completion
hold on
% The next two plots should be identical
% plot(Nj,0.3*sig_bar(Nj,(5/N)))
% stairs(Nj,sig_bar(Nj -log(0.3)/(5/N),(5/N)))
bsig = [];
for i=1:N
    bsig(i,:) = bsigma_0(i)*sig_bar(episodes,(tau/c0));
    indx(i) = find(bsig(i,:)<0.01,1,'first');
    disp(episodes(indx(i)))
    p = plot(episodes,bsigma_0(i)*sig_bar(episodes,(tau/c0)));
    semilogy(episodes(indx(i))*ones(numel(0:0.00001:1),1),0:0.00001:1,'--','Color',p.Color);
end

% Effects of changing the rate
% stairs(Nj,sig_bar(Nj,1.2*alpha))
xlabel('Episodes [n]','FontSize',25)

%%

sig_bar = @(Nj,alpha) exp(-alpha*Nj);
close all
semilogy(Nj,sig_bar(Nj,(5/N)))
hold on
plot(Nj,0.01*ones(size(Nj)),'k--')
semilogy(Nj,0.3*sig_bar(Nj,(5/N)))
semilogy(Nj,sig_bar(Nj -log(0.3)/(5/N),(5/N)))
semilogy(Nj,sig_bar(Nj,0.3))
%%
clc
close all

x = ode4(@(t,x) -A(1,1)*x,n,1);
figure
stairs(n,x)
xlabel('Episodes [n]','FontSize',25)

%% ************************************************************************

clearvars
close all
clc
clc
c0      = 100;
alpha_i = 10;
N_S     = 6;
N_K     = 2;
N       = N_S/N_K;
Spp     = sympositivedefinitefactory(N);

A     = Spp.rand();


% A     = A + abs(min(A,[],'all'));
% A     = A./(sum(A,'all'))

% Alternative construction for A
A = rand(N); % generate a random n x n matrix
A = 0.5*(A+A');
A = A.*(~eye(N)) + N*eye(N);


A
[V,D] = eig(-A)
c     = (V\ones(N,1));
%
close all
n = 0:0.001:c0;
% n=0;
f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
x = f(n);

f_i = @(t,x) -A.*eye(N)*x; 
f_c = @(t,x) -A*x; 
% x = ode4(f,n,sum(A,2));
x_i = ode4(f_i,n,ones(N,1));
x_c  = ode4(f_c,n,ones(N,1));


min(x_c(:))
figure('Color',[1 1 1])
plot(n,x_i,'k','LineWidth',1.5)
hold on
% plot(n,mean(x,2),'k--')
plot(n,x_c)
xlabel('Episodes n','FontSize',25)
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
xlim([0 5])
%%
close all
 plot(n,exp(-3*n) + exp(-8*n))
 plot(n,0.7*exp(-3*n))
 hold on
 plot(n,0.3*exp(-8*n))
 plot(n,0.7*exp(-3*n) + 0.3*exp(-8*n),'k--')
 xlim([0 10])
%% ************************************************************************
% SIMULATION OF THE COMPLETE SYSTEM
% *************************************************************************

clearvars
close all
clc
clc
c0      = 100;
alpha_i = 10;
N_S     = 4;
N_K     = 2;
N       = N_S/N_K;

Spp   = sympositivedefinitefactory(N);
A1     = Spp.rand();
A1     = A1 + abs(min(A1,[],'all'));

Spp   = sympositivedefinitefactory(N);
A2     = Spp.rand();
A2     = A2 + abs(min(A2,[],'all'));

B = 0.1/3*ones(N);

A = [A1, B; B, A2]

[V,D] = eig(-A/c0)
% c     = (V\ones(N,1));
%
close all
n = 0:0.001:c0;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
% f = @(t,x) -eye(size(A))*x; 
% x = ode4(f,n,sum(A,2));
x = ode4(f,n,ones(size(A,1),1));

min(x(:))
figure('Color',[1 1 1])
plot(n,x(:,1:2),'b--')
hold on
plot(n,x(:,3:4),'r')
% plot(n,sum(x,2),'k:')
xlabel('Episodes n','FontSize',25)
xlim([0 5])
ylabel('$\bar{\sigma}^{(C)}_{j,k}$','Interpreter','latex','FontSize',25)

h = zeros(N_K, 1);
h(1) = plot(NaN,NaN,'--b');
h(2) = plot(NaN,NaN,'-r');
leg = legend(h, 'k_1','k_1');
set(leg,'FontSize',15)

%% ************************************************************************
% SIMULATION WITH PRE DEFINED EIGENVECTORS
% *************************************************************************

clc
clearvars

% V = [0.2 0.1; 0.6 0.1];
% W = [0.2 0.1;0.1 0.6];
N = 3;
Spp   = sympositivedefinitefactory(N);
W     = Spp.rand();
W     = W + abs(min(W,[],'all'));
W     = W.*(~eye(N)) + eye(N);
W     = W./sum(W,2)

% W = [0.7 0.3;0.3 0.7];
% D = diag([5 5]);
D = diag(rand(N,1));%5*eye(N);
A = W*D*(W^-1)
[V_A,D_A] = eig(A)
c  = (W\sum(W,2));
%
close all
n = 0:0.01:50;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
x = ode4(f,n,sum(W,2));



min(x(:))
plot(n,x)
hold on
plot(n,mean(x,2),'k--')

%% ************************************************************************
% SIMULATION OF THE DIFFUSION PROCESS ON A GRAPH
% *************************************************************************
clearvars
clc
close all

N          = 3;
Spp        = sympositivedefinitefactory(N);
A          = Spp.rand();
A          = A + abs(min(A,[],'all'));
A          = A.*(~eye(size(A)))
D          = diag(sum(A,2))
L          = D - A
[U,lambda] = eig(L)

close all
n = 0:0.001:100;
f = @(t,x) -L*x; 
% x = ode4(f,n,sum(A,2));
% x = ode4(f,n,ones(N,1));
x0 = rand(N,1);
x0 = N*(x0/sum(x0));
x = ode4(f,n,x0);
plot(n,x)
xlabel('Episodes [n]','FontSize',25)
% xlim([0 10])
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
