%% - Functions folder in the same folder
clc
clearvars

cd(fileparts(matlab.desktop.editor.getActiveFilename)); % use this to CD
addpath(genpath(pwd))

% Threshold to consider a skill learned
epsilon = 0.01;

% Fundamental complexity (AKA max number of episodes to learn any skill)
c0 = 100;
% Trial episodes
episodes  = 0:0.01:c0; 

% Total number of skills
totalSkills = 8*64;

% Number of clusters
totalClusters = 4;

% Number of skills per cluster
skillsPerCluster = totalSkills/totalClusters;

% Base learning rate
alpha_min = -log(epsilon)/c0;%0.05;
alpha_max = 1.5*alpha_min;

delta = -log(epsilon)/skillsPerCluster;%0.05;

eta_0   = 0.1;

f = @(eta, N_zeta) eta.*N_zeta+1;
g = @(delta, N_zeta) exp(-delta*N_zeta);

%% number of robots available
maxNumberOfRobots = 128;
totalSimulationScenarios = 5;


%% **************************************************************************
% Dynamics of ISOLATED LEARNING 
% *************************************************************************
clc
close all
rng('default')
clear c_jk_iso_episodes
for scenario = 1:totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]
        fig       = figure('color','w');
        % Transfer learning factor
        beta_k   = 0;
        c_jk_iso = zeros(skillsPerCluster, totalClusters);
        % Loop over clusters
        for k = 1:totalClusters
            % disp(['Running Scenario: Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            formatSpec = 'Scenario: %4i | Cluster: %4i | No. Robots: %4i\n';
            fprintf(formatSpec,scenario, k, numberOfRobots)
            subplot(1,totalClusters,k)
            % Loop over skills
            for learnedSkills = 0:(skillsPerCluster/numberOfRobots)-1
                j          = learnedSkills+1;
                sigmaBar_0 = ones(numberOfRobots,1);
                % a          = -alpha_min;
                % F          = a*eye(numberOfRobots);
                a          = -(alpha_min + (alpha_max-alpha_min).*rand(numberOfRobots,1));
                F          = diag(a);
                remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, episodes, sigmaBar_0));
                loglog(episodes,remainingKnowledge,'k-','LineWidth',1)
                hold on
                % c_jk_iso(j, k) = -log(epsilon)/alpha_min;
                c_jk_iso(j, k) = ceil(mean(episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<epsilon)),1:numberOfRobots,'UniformOutput',false)))));
            end
            p = area(episodes, epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(k),'})$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{IsL})}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
            xlim([0 episodes(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
        end
        % legend(skill_labels)
        for ax=1:totalClusters
            fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
        end
        fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
    %     p_aux = plot(NaN, NaN);
    %     legend(p_aux,['m = ', num2str(m)]);
        pause(1)
        tightfig(fig);
    
        SAVE_FIG = 0;
        if numberOfRobots == 32 && SAVE_FIG == 1
            exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_isolated_learning.png'),'Resolution',600);
            close(gcf);
        else 
            close(gcf);
        end
    
        c_jk_iso_episodes(robotBatchIndex,:,scenario) = [sum(c_jk_iso,1), sum(c_jk_iso,'all')] ;
        robotBatchIndex      = robotBatchIndex+1; 
    end
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
for scenario = 1:totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]
        fig     = figure('color','w');
        % Self learning factor
        Alpha  = diag(alpha_min + (alpha_max-alpha_min).*rand(numberOfRobots,1)); 
        % Transfer learning factor
        beta_k  = 0;
        c_jk_il = zeros(skillsPerCluster,totalClusters);
        % Loop over clusters
        for k = 1:totalClusters
            %disp(['Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            formatSpec = 'Scenario: %4i | Cluster: %4i | No. Robots: %4i\n';
            fprintf(formatSpec,scenario, k, numberOfRobots)
            subplot(1,totalClusters,k)
            % Loop over skills
            for learnedSkills = 0:(skillsPerCluster/numberOfRobots)-1
                j          = learnedSkills+1;
                sigmaBar_0 = (1-beta_k)*g(delta, learnedSkills).*ones(numberOfRobots,1);
                % a          = -alpha_min*((1-beta_k)^(-1))*f(eta_0, learnedSkills);
                % F          = a*eye(numberOfRobots);
                eta_robots = abs(0.1*eta_0.*randn(numberOfRobots,1) + eta_0);
                F          = -Alpha.*((1-beta_k)^(-1)).*f(eta_robots, learnedSkills);
                remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, episodes, sigmaBar_0));
                loglog(episodes,remainingKnowledge,'k-','LineWidth',1)
                hold on
                %c_jk_il(j, k) = -(log(epsilon) + delta*learnedSkills)/(alpha_min*f(eta_0, learnedSkills));
                c_jk_il(j, k) = ceil(mean(episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<epsilon)),1:numberOfRobots,'UniformOutput',false)))));
            end
            disp(['Skills seen:' num2str(j)]);
            p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(k),'})$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{IL})}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
            xlim([0 episodes(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
        end
        % legend(skill_labels)
        for ax=1:totalClusters
            fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
        end
        fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
        pause(1)
        tightfig(fig);
    
        SAVE_FIG = 0;
        if numberOfRobots == 32 && SAVE_FIG == 1
            exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_incremental_learning.png'),'Resolution',600)
            close(gcf);
        else
            close(gcf);
        end
    
        c_jk_il_episodes(robotBatchIndex,:,scenario) = [sum(c_jk_il,1), sum(c_jk_il,'all')];
        robotBatchIndex = robotBatchIndex + 1;
    end
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
for scenario = 1:totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]
        fig       = figure('color','w');
        % Self learning factor
        Alpha  = diag(alpha_min + (alpha_max-alpha_min).*rand(numberOfRobots,1));         
        c_jk_itl  = zeros(skillsPerCluster, totalClusters);
        % Loop over clusters
        for k = 1:totalClusters
            % disp(['Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            formatSpec = 'Scenario: %4i | Cluster: %4i | No. Robots: %4i\n';
            fprintf(formatSpec,scenario, k, numberOfRobots)        
            subplot(1,totalClusters,k)
            % Transfer learning factor
            % beta_k = (k-1)/totalClusters;
            beta_k = (k-1)/totalClusters.* rand(numberOfRobots,1);
            % Loop over skills
            for learnedSkills = 0:(skillsPerCluster/numberOfRobots)-1
                j          = learnedSkills+1;
                sigmaBar_0 = (1 - beta_k).*g(delta, learnedSkills).*ones(numberOfRobots,1);
                % a          = -alpha_min*((1 - beta_k)^(-1))*f(eta_0, learnedSkills);
                % F          = a*eye(numberOfRobots);
                eta_robots = abs(0.1*eta_0.*randn(numberOfRobots,1) + eta_0);
                F          = -Alpha.*((1-beta_k).^(-1)).*f(eta_robots, learnedSkills);
                remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, episodes, sigmaBar_0));    
                loglog(episodes,remainingKnowledge,'k-','LineWidth',1)
                ylim([-0.1 1.1])
                hold on
                %c_jk_itl(j, k) = -(log(epsilon*(1 - beta_k).^(-1)) + delta*learnedSkills)/(alpha_min*(1 - beta_k)^(-1)*f(eta_0, learnedSkills));
                c_jk_itl(j, k) = ceil(mean(episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<epsilon)),1:numberOfRobots,'UniformOutput',false)))));
            end
            p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(k),'})$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{TIL})}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
            xlim([0 episodes(end)])
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
    
        SAVE_FIG = 0;
        if numberOfRobots == 32 && SAVE_FIG == 1
            exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_incremental_transfer_learning.png'),'Resolution',600)
            close(gcf);
        else
            close(gcf);
        end    
    
        c_jk_til_episodes(robotBatchIndex,:,scenario) = [sum(c_jk_itl,1), sum(c_jk_itl,'all')] ;
        robotBatchIndex                      = robotBatchIndex + 1; 
    end
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

warning("<<ALL>> m robots placed in one cluster at a time concurrently exchanging knowledge")
pause(2)
clear c_jk_cl_episodes
for scenario = 1:totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]   
        Alpha   = diag(alpha_min + (alpha_max-alpha_min).*rand(numberOfRobots,1)); 
        A_full  = ones(numberOfRobots) - eye(numberOfRobots);
        A       = A_full;
        fig     = figure('color','w');
        
        % Coupling
        gamma_0 = 0.05;
        Gamma   = 0.25*eta_0.*randn(numberOfRobots,numberOfRobots) + gamma_0;
        Gamma   = (Gamma + transpose(Gamma))./2;
        Gamma   = Gamma - diag(Gamma).*eye(numberOfRobots);
        
        r       = double(gamma_0==0) + double(gamma_0~=0)*numberOfRobots;
        c_jk_cl = zeros(skillsPerCluster, totalClusters);
        % Loop over clusters
        for k = 1:totalClusters
            disp(['Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            subplot(1,totalClusters,k)
            beta_k = (k-1)/totalClusters.*rand(numberOfRobots,1);
            % Loop over skills
            for learnedSkills = 0:skillsPerCluster/numberOfRobots-1    
                j          = learnedSkills+1;
                sigmaBar_0 = (1 - beta_k).*g(delta, r*learnedSkills).*ones(numberOfRobots,1);
                eta_robots = abs(0.1*eta_0.*randn(numberOfRobots,1) + eta_0);
                h          = -Alpha.*((1 - beta_k).^(-1)).*f(eta_robots, r*learnedSkills).*eye(numberOfRobots);
                % F        = a*eye(m) + gamma.*A;
                % lambda_F = eig(F);
                % disp(lambda_F)
                % if any(real(lambda_F)>0)
                %     warning('Unstable!!!')
                %     return
                % end
                %sigmaBar   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, n, sigmaBar_0));
                % remainingKnowledge   = transpose(ode4(@(n,sigmaBar) fcn_knowledge_integration(sigmaBar, numberOfRobots, diag(a), gamma_0.*A), episodes, sigmaBar_0));
                remainingKnowledge   = transpose( ...
                    ode4(@(n,sigmaBar) fcn_collective_knowledge_dynamics(sigmaBar, numberOfRobots, h, Gamma), episodes, sigmaBar_0));
                %c_jk_cl(j, k)  = episodes(min(find(remainingKnowledge(1,:)<epsilon)));
                c_jk_cl(j, k) = ceil(mean(episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<epsilon)),1:numberOfRobots,'UniformOutput',false)))));
                %semilogy(episodes,remainingKnowledge,'LineWidth',2)
                % semilogy(episodes,remainingKnowledge,'k-','LineWidth',1)
                loglog(episodes,mean(remainingKnowledge,1),'k-','LineWidth',1)
                loglog(episodes,remainingKnowledge,'k-','LineWidth',1)
                % plot(c_jk_cl(j, k),max(mean(remainingKnowledge,1)),'ko','LineWidth',1)
                % plot(episodes,remainingKnowledge,'LineWidth',2)
                xlim([0 episodes(end)])
                ylim([1E-3 1E0])
                % xticks([0 50 100])
                xticks([10^0 10^1 10^2])
                xticklabels({'1','10', '$c_0$'})
                yticks([1E-3 1E-2 1E-1 1E0])
                yticklabels({'','$\epsilon$', '', '1'})
                set(gca,'TickLabelInterpreter','latex')
                hold on
            end
            p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(k),'}$)'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CL})}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
        end
        for ax=1:totalClusters
            fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
        end
        fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
        pause(1)
        tightfig(fig);
        
    
        SAVE_FIG = 0;
        if numberOfRobots == 32 && SAVE_FIG == 1
            exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_collective_learning.png'),'Resolution',600)
            close(gcf);
        else
            close(fig)
        end        
    
        c_jk_cl_episodes(robotBatchIndex,:,scenario) = [sum(c_jk_cl,1), sum(c_jk_cl,'all')] ;
        robotBatchIndex                     = robotBatchIndex + 1; 
    end
end
%% Alternative robot distribution: EQUAL robots per cluster
close all

warning("<<EQUAL>> number of robots per cluster")
pause(2)
clear c_jk_cl_dist_episodes
for scenario = 1:totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    % * NOTE: index starts in 4 to have at least one robot per cluster
    for numberOfRobots = [4,8,16,32,64,128]
        Alpha  = diag(alpha_min + (alpha_max-alpha_min).*rand(numberOfRobots,1));
        A_full = ones(numberOfRobots) - eye(numberOfRobots);
        A      = A_full;
    
    
        % % Inter cluster scaling matrix
        % %beta_k = 1/N_K;
        % beta_k = 1/(numberOfRobots*totalClusters);
        % B      = beta_k*A;
        % 
        % % Robots concurrently exchanging knowledge
        % r = numberOfRobots;
        
        clc
    
        % beta_k = (1/totalClusters);
        % B      = beta_k*A;
        % for i=1:totalClusters
        %    B((numberOfRobots/totalClusters)*(i-1) + 1:(numberOfRobots/totalClusters)*i,(numberOfRobots/totalClusters)*(i-1) + 1:(numberOfRobots/totalClusters)*i) = ones(numberOfRobots/totalClusters) - eye(numberOfRobots/totalClusters);
        % end
        % B = triu(B) + transpose(triu(B));
        
        fig = figure('color','w');
        c_jk_cl_dist  = zeros(totalSkills/numberOfRobots, 1);%zeros(N_Z, N_K);
        
        % Coupling
        gamma_0 = 0.05;
        Gamma   = 0.25*eta_0.*randn(numberOfRobots,numberOfRobots) + gamma_0;
        Gamma   = (Gamma + transpose(Gamma))./2;
        Gamma   = Gamma - diag(Gamma).*eye(numberOfRobots);
        r       = double(gamma_0==0) + double(gamma_0~=0)*numberOfRobots;
        
    %     for k = 1:(N_S/m)  
    %         disp(['Batch ',num2str(k),' -------------------------'])
    %     %     subplot(2,N_K,k)        
    %         N_zeta   = k-1;
    %         sigmaBar_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
    %         a        = -alpha*f(eta, r*N_zeta);%*(1- beta_k)^(-1);
    %         F        = a*eye(m) + gamma*A.*B;
    %         lambda_F = eig(F);
    %         sigmaBar   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, n, sigmaBar_0));
    %         c_jk_cl_dist(k) = n(min(find(sigmaBar(1,:)<epsilon)));
    %         disp(lambda_F)        
    %         semilogy(n,sigmaBar,'LineWidth',2)
    %         xlim([0 n(end)])
    %         ylim([1E-3 1E0])
    %         xticks([0 50 100])
    %         xticklabels({'0','', '$c_0$'})
    %         yticks([1E-50 1E-2 1E-1 1E0])
    %         yticklabels({'','$\epsilon$', '', '1'})
    %         set(gca,'TickLabelInterpreter','latex')
    %         hold on
    %     end
        for learnedSkills = 0:(totalSkills/numberOfRobots)-1
            % Beta adapts at every cycle
            % ============================================
            beta_k = r*learnedSkills/totalSkills;%.*rand(numberOfRobots,1); % From each cluster, only this fraction of knowlede can be transfered
    %         beta_k = (r*N_zeta/N_Z)*(1/N_K)%min(r*N_zeta/N_Z,1)*(1/N_K)
            B      = beta_k*A;
            for i=1:totalClusters
               B((numberOfRobots/totalClusters)*(i-1) + 1:(numberOfRobots/totalClusters)*i,(numberOfRobots/totalClusters)*(i-1) + 1:(numberOfRobots/totalClusters)*i) = ones(numberOfRobots/totalClusters) - eye(numberOfRobots/totalClusters);
            end
            B = triu(B) + transpose(triu(B));
            perturbationB = rand(numberOfRobots,numberOfRobots);
            perturbationB(B==0) = 0;
            perturbationB(B==1) = 0;
            % Ensure the matrix is symmetric
            perturbationB   = (perturbationB + transpose(perturbationB))./2;
            % B = B.*perturbationB;
            if sum(B-B','all') ~= 0
                warning('Error in B matrix')
            end
            % ============================================
    
    
            j        = learnedSkills + 1;
            disp(['Skills batch: ',num2str(j),' -------------------------'])
            sigmaBar_0 = (1 - beta_k)*g(delta, r*learnedSkills).*ones(numberOfRobots,1);
            eta_robots = abs(0.1*eta_0.*randn(numberOfRobots,1) + eta_0);
            h          = -Alpha.*f(eta_robots, r*learnedSkills).*eye(numberOfRobots);        
            %sigmaBar   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, n, sigmaBar_0));
            remainingKnowledge   = transpose(ode4(@(n,sigmaBar) fcn_collective_knowledge_dynamics(sigmaBar, numberOfRobots, h, Gamma.*B), episodes, sigmaBar_0));
            % c_jk_cl_dist(j) = episodes(min(find(remainingKnowledge(1,:)<epsilon)));
            % loglog(episodes,remainingKnowledge,'k-',LineWidth=1)
    
            c_jk_cl_dist(j) = ...
                ceil(mean(episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<epsilon)),1:numberOfRobots,'UniformOutput',false)))));
                    loglog(episodes,mean(remainingKnowledge,1),'k-','LineWidth',1)
    
            xlim([0 episodes(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-50 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
            hold on
        end
    
        p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
        plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
        % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
        xlabel('Episodes','FontSize',25)
        ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CL})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
        fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 18/2, 3, 1)
        pause(1)
        tightfig(fig);
        c_jk_cl_dist_episodes(robotBatchIndex, scenario) = sum(c_jk_cl_dist);
        robotBatchIndex                        = robotBatchIndex + 1; 
    end
end
%% Total number of episodes
complexities = {c_jk_iso_episodes,c_jk_il_episodes,c_jk_til_episodes,c_jk_cl_episodes,c_jk_cl_dist_episodes};

cmap = colororder();
clc
close all
fig = figure('color','w');
p = NaN(1,5);
hold on
for index = 1:5
    if index<5
        the_mean   = ceil(mean(complexities{index},3));
        the_std    = ceil(std(complexities{index},[],3));
        uppeBound  = the_mean(:,end) + the_std(:,end);
        lowerBound = the_mean(:,end) - the_std(:,end);
        x = 1:7;
        patch([x fliplr(x)], [lowerBound'  fliplr(uppeBound')], [0.6  0.7  0.8],'FaceColor',cmap(index,:),'FaceAlpha',0.25,'EdgeColor','w');       
    else
        the_mean   = ceil(mean(complexities{index},2));
        the_std    = ceil(std(complexities{index},[],2));
        uppeBound  = the_mean + the_std;
        lowerBound = the_mean - the_std;
        x = 2:7;
        patch([x fliplr(x)], [lowerBound'  fliplr(uppeBound')], [0.6  0.7  0.8],'FaceColor',cmap(index,:),'FaceAlpha',0.25,'EdgeColor','w');
    end
    % p = [p, plot(x, the_mean, 'o-', 'LineWidth', 1,'Color',cmap(index,:),'MarkerFaceColor',cmap(index,:))];        
    p(index) = plot(x, the_mean(:,end), 'o-', 'LineWidth', 1,'Color',cmap(index,:),'MarkerFaceColor',cmap(index,:));        
end
plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
xlabel('Number of robots','FontSize',25)
ylabel('Complexity [episodes]','FontSize',25)
leg = legend(p,'IsL','IL','TIL','CL','CL$_\mathrm{distributed}$');
fcn_scrpt_prepare_graph_science_std(fig, gca, p, leg, [], 18/2, 3, 1)
leg.Location = 'northeast';
leg.Box = 'on';
% fig = gcf;           % generate a figure
% tightfig(fig);
box on
set(gca, 'YScale', 'log')
pause(1)

SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_episodes_per_n_robots_v1.png'),'Resolution',600)
    close(gcf);
end  
% set(gca, 'YScale', 'log')
%%
clc
close all
y = rand(1,10); % your mean vector;
x = 1:numel(y);
std_dev = 1;
uppeBound = y + std_dev;
lowerBound = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [uppeBound, fliplr(lowerBound)];
fill(x2, inBetween, 'g');
hold on;
plot(x, y, 'r', 'LineWidth', 2);

%% Total number of episodes
close all
clc

c_jk_cl_episodes_mean = c_jk_cl_episodes(:,end,3);
c_jk_cl_episodes_std = ceil(std(c_jk_cl_episodes,[],3));
x = 1:7;
uppeBound = c_jk_cl_episodes_mean(:,end) + c_jk_cl_episodes_std(:,end);
lowerBound = c_jk_cl_episodes_mean(:,end) - c_jk_cl_episodes_std(:,end);
x2 = [x, fliplr(x)];
inBetween = [uppeBound', fliplr(lowerBound)'];
fill(x2, inBetween, 'g');
hold on;

%%
close all
patch([x fliplr(x)], [lowerBound'  fliplr(uppeBound')], [0.6  0.7  0.8],'FaceColor',[0.5 0.5 0.5],'FaceAlpha',0.25,'EdgeColor','w');
hold on
plot(x, c_jk_cl_episodes_mean(:,end), 'k', 'LineWidth', 1);
% set(gca, 'YScale', 'log')


%%

% factor = [2,4,8,16,32,64,128]'*105E3;
fig = figure('color','w');
% p0 = semilogy([2,4,8,16,32,64,128],c_jk_iso_episodes(:,end),'o-');
% hold on
% p1 = plot([2, 4, 8, 16, 32, 64, 128],c_jk_il_episodes(:,end),'o-');
% p2 = plot([2, 4, 8, 16, 32, 64, 128],c_jk_til_episodes(:,end),'o-');
% p3 = plot([2, 4, 8, 16, 32, 64, 128],c_jk_cl_episodes(:,end),'o-');
% p4 = plot(   [4, 8, 16, 32, 64, 128],c_jk_cl_dist_episodes,'o-');
p0 = semilogy(1:7,ceil(c_jk_iso_episodes(:,end)),'o-');
hold on
p1 = plot(1:7,ceil(c_jk_il_episodes(:,end)),'o-');
p2 = plot(1:7,ceil(c_jk_til_episodes(:,end)),'o-');
p3 = plot(1:7,c_jk_cl_episodes(:,end),'o-');
p4 = plot(2:7,c_jk_cl_dist_episodes,'o-');

plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)
% xticks([2,4,8,16,32,64,128])
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
% xlabel('m','FontSize',25)
% ylabel('$C_\mathcal{S}$','FontSize',25,'Interpreter','latex')
xlabel('Number of robots','FontSize',25)
ylabel('Complexity [episodes]','FontSize',25)
% leg = legend('IsL','IL','TIL','CL','CL2');
leg = legend('IsL','IL','TIL','CL','CL$_\mathrm{distributed}$');
fcn_scrpt_prepare_graph_science_std(fig, gca, [p0, p1, p2, p3, p4], leg, [], 18/2, 3, 1)
leg.Location = 'northeast';
leg.Box = 'on';
fig = gcf;           % generate a figure
tightfig(fig);
pause(1)

SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_episodes_per_n_robots.png'),'Resolution',600)
    close(gcf);
end  
%%
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','total_energy_per_n_robots'));
disp('Printing done!')
%%

y = rand(1,10); % your mean vector;
x = 1:numel(y);
std_dev = 1;
uppeBound = y + std_dev;
lowerBound = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [uppeBound, fliplr(lowerBound)];
fill(x2, inBetween, 'g');
hold on;
plot(x, y, 'r', 'LineWidth', 2);
%% Total energetic cost
close all

%factor = [2,4,8,16,32,64,128]'*105E3;
factor = transpose([2,4,8,16,32,64,128])*(40 + 300 + 1416)*(60);
fig = figure('color','w');
p0 = semilogy([2,4,8,16,32,64,128],factor.*c_jk_iso_episodes(:,end),'o-');
hold on
p1 = plot([2, 4, 8, 16, 32, 64, 128],factor.*c_jk_il_episodes(:,end),'o-');
p2 = plot([2, 4, 8, 16, 32, 64, 128],factor.*c_jk_til_episodes(:,end),'o-');
p3 = plot([2, 4, 8, 16, 32, 64, 128],factor.*c_jk_cl_episodes(:,end),'o-');
p4 = plot(   [4, 8, 16, 32, 64, 128],factor(2:end).*c_jk_cl_dist_episodes','o-');
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
SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_energy_per_n_robots.png'),'Resolution',600)
    close(gcf);
end    
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


r        = numberOfRobots;
c_jk_cl  = zeros(skillsPerCluster/numberOfRobots, totalClusters);
kappa = zeros(1,skillsPerCluster+1);
for k = 1:totalClusters
    disp(['Cluster ',num2str(k),' -------------------------'])
    subplot(1,totalClusters,k)
    beta_k = (k-1)/totalClusters;
    for learnedSkills = 0:skillsPerCluster/numberOfRobots-1    
        j        = learnedSkills+1;
        sigmaBar_0 = (1- beta_k)*g(delta, r*learnedSkills).*ones(numberOfRobots,1);
        a        = -alpha_max*f(eta_0, r*learnedSkills)*(1- beta_k)^(-1);
        gamma_0    = 0.005;%-0.3*(-(a)/max(eig(abs(A))));
        F        = a*eye(r) + gamma_0*A;
        lambda_F = eig(F)
        remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, episodes, sigmaBar_0));
%         for i=0:m-1
%             c_jk_cl(j+i, k) = n(min(find(sigmaBar(i+1,:)<epsilon))); 
%         end
%         for r_i=1:r
%             c_jk_cl(r_i, k) = n(min(find(sigmaBar(r_i,:)<epsilon)));
%         end
        c_jk_cl(j, k) = episodes(min(find(remainingKnowledge(1,:)<epsilon)));
        semilogx(episodes,remainingKnowledge,'LineWidth',3)
        hold on
%         kappa(N_zeta + 1)  = sum(sigmaBar);
    end
    p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 episodes(end)])
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
for k = 1:totalClusters
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
        var = (cycle-1)*numberOfRobots;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha_max, s, c0, tau, epsilon, learning_type_1), ...
                                             episodes, ...
                                             1-(cycle-1)/(N/numberOfRobots)*ones(numberOfRobots,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(episodes,x,'LineWidth',2);
    hold on
end
plot(episodes,exp(A_i(1,1)*episodes),'k--','LineWidth',2)
p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 episodes(end)])
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
numberOfRobots = 2;

% Skill index
s         = transpose(1:N);

% Skill vector
Nj        = 0:N;

% Rate at which knowledge is depleted per skill acquired
% *NOTE: determined to satisfy <threshold = exp(-alpha*N)>
alpha_max     = -log(epsilon)/N;  % log(0.01)/N;%(rate/N);

% Initial values of the subsequent skills based on alpha
sigmaBar_0  = exp(-alpha_max*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:c0*N;
episodes  = 0:0.01:100; 

% Initial complexity
c0        = 100;

% Initial knowledge aquisition rate per episode
% *NOTE: determined to satisfy <threshold = exp(-(rate/c0)*c0)>
tau       = -log(epsilon)*ones(numberOfRobots,1);
A_i       = -diag(tau/c0);

close all
%theta          = 1:15;%&10*rand(N*(N+1)/2,1);
% theta          = normrnd(5*ones(N*(N+1)/2,1),0.33*ones(N*(N+1)/2,1));
theta          = 1*ones(numberOfRobots*(numberOfRobots+1)/2,1);
% theta(N+1:end) = rand(N*(N+1)/2 - N,1);
gamma_0           = 1;%rand(1);
theta(numberOfRobots+1:end) = gamma_0*ones(numberOfRobots*(numberOfRobots+1)/2 - numberOfRobots,1);
ind            = find(triu(ones(numberOfRobots,numberOfRobots),1));
L              = zeros(numberOfRobots) + diag(theta(1:numberOfRobots));
L(ind)         = theta(numberOfRobots+1:end);
G              = L + triu(L,1)';
eig(G);
% G = (L')*L


X = NaN(numberOfRobots,numel(episodes),N/numberOfRobots);
fig = figure('Color','w');
fig.WindowState = 'maximized';
for cycle = 1:N/numberOfRobots
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
              fcn_general_dynamics(t, x, var*A, alpha_max, s, c0, tau, epsilon, learning_type_1), ...
                                             episodes, ...
                                             ones(numberOfRobots,1));        
    else
        var = (cycle-1)*numberOfRobots;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha_max, s, c0, tau, epsilon, learning_type_1), ...
                                             episodes, ...
                                             1-(cycle-1)/(N/numberOfRobots)*ones(numberOfRobots,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(episodes,x,'LineWidth',2);
    hold on
end
plot(episodes,exp(A_i(1,1)*episodes),'k--','LineWidth',2)
p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 episodes(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')

%%
% -------------------------------------------------------------------------
learning_type_2 = 'collective';
% (t, x, A, alpha, s, c0, rate, threshold, type)
x_nonebounded = ode4(@(t,x) ...
      fcn_general_dynamics(t, x, A, alpha_max, s, c0, tau, epsilon, learning_type_2), ...
                                     episodes, ...
...
ones(N,1));

figure('Color','w')
p = plot(episodes,x_nonebounded,':','LineWidth',1.5);
hold on
set(gca,'ColorOrderIndex',1)
p = plot(episodes,x,'LineWidth',2);
plot(episodes,exp(A_i(1,1)*episodes),'k--','LineWidth',2)
p = area(episodes,  epsilon*ones(numel(episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(episodes,epsilon*ones(numel(episodes),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 episodes(end)])
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
alpha_max     = -log(epsilon)/N;%-log(0.01)/N;%(rate/N);
% alpha     = (rate/N);
sigmaBar_0  = exp(-alpha_max*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:c0*N;
episodes  = 0:0.01:1000; 
learning_type = 'incremental_3';
x = ode4(@(t,x) fcn_general_dynamics(t, x, A, alpha_max, s, c0, tau, epsilon,...
                learning_type), ...
                                     episodes, ...
                                     zeros(N,1) +1*sigmaBar_0(1:end-1)');
x = x';
figure('Color','w')
if strcmp(learning_type,'incremental_2')
%     p = plot(n,sigmaBar_0(1:end-1)'-x,'LineWidth',2);
    p = plot(episodes,x,'LineWidth',2);
else
    p = plot(episodes, x,'LineWidth',2);
end
hold on

% p = plot(n,(1-sigmaBar_0(1:end-1))' - x,'LineWidth',2);
% p = plot(n, - x,'LineWidth',2);
% hold on
plot(episodes(end),sigmaBar_0(1:end-1),'*')
xlabel('Episodes [n]','FontSize',25)
ylabel('Knowledge ','FontSize',25)
% xlim([0 sum(c0*sigmaBar_0)])
ylim([0 1])
%
title('Incremental')

% clear indx
% for j = 1:size(x,1)
%     delta  = x(j,:) - sigmaBar_0(j);
%     indx(j) = find(abs(delta) < threshold,1,'first');
% %     disp(n(indx(j)))
% end
% figure('Color','w')
% plot(1:N,c0*sigmaBar_0(1:end-1),'--','LineWidth',2)
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
    bsig(i,:) = sigmaBar_0(i)*sig_bar(episodes,(tau/c0));
    indx(i) = find(bsig(i,:)<0.01,1,'first');
    disp(episodes(indx(i)))
    p = plot(episodes,sigmaBar_0(i)*sig_bar(episodes,(tau/c0)));
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

x = ode4(@(t,x) -A(1,1)*x,episodes,1);
figure
stairs(episodes,x)
xlabel('Episodes [n]','FontSize',25)

%% ************************************************************************

clearvars
close all
clc
clc
c0      = 100;
alpha_i = 10;
totalSkills     = 6;
totalClusters     = 2;
N       = totalSkills/totalClusters;
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
episodes = 0:0.001:c0;
% n=0;
f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
x = f(episodes);

f_i = @(t,x) -A.*eye(N)*x; 
f_c = @(t,x) -A*x; 
% x = ode4(f,n,sum(A,2));
x_i = ode4(f_i,episodes,ones(N,1));
x_c  = ode4(f_c,episodes,ones(N,1));


min(x_c(:))
figure('Color',[1 1 1])
plot(episodes,x_i,'k','LineWidth',1.5)
hold on
% plot(n,mean(x,2),'k--')
plot(episodes,x_c)
xlabel('Episodes n','FontSize',25)
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
xlim([0 5])
%%
close all
 plot(episodes,exp(-3*episodes) + exp(-8*episodes))
 plot(episodes,0.7*exp(-3*episodes))
 hold on
 plot(episodes,0.3*exp(-8*episodes))
 plot(episodes,0.7*exp(-3*episodes) + 0.3*exp(-8*episodes),'k--')
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
totalSkills     = 4;
totalClusters     = 2;
N       = totalSkills/totalClusters;

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
episodes = 0:0.001:c0;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
% f = @(t,x) -eye(size(A))*x; 
% x = ode4(f,n,sum(A,2));
x = ode4(f,episodes,ones(size(A,1),1));

min(x(:))
figure('Color',[1 1 1])
plot(episodes,x(:,1:2),'b--')
hold on
plot(episodes,x(:,3:4),'r')
% plot(n,sum(x,2),'k:')
xlabel('Episodes n','FontSize',25)
xlim([0 5])
ylabel('$\bar{\sigma}^{(C)}_{j,k}$','Interpreter','latex','FontSize',25)

h = zeros(totalClusters, 1);
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
episodes = 0:0.01:50;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
x = ode4(f,episodes,sum(W,2));



min(x(:))
plot(episodes,x)
hold on
plot(episodes,mean(x,2),'k--')

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
episodes = 0:0.001:100;
f = @(t,x) -L*x; 
% x = ode4(f,n,sum(A,2));
% x = ode4(f,n,ones(N,1));
x0 = rand(N,1);
x0 = N*(x0/sum(x0));
x = ode4(f,episodes,x0);
plot(episodes,x)
xlabel('Episodes [n]','FontSize',25)
% xlim([0 10])
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
