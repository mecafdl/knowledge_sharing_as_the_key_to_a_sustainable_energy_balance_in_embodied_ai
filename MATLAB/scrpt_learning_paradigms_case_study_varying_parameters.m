%% - Functions folder in the same folder
clc
clearvars

cd(fileparts(matlab.desktop.editor.getActiveFilename)); % use this to CD
addpath(genpath(pwd))

% Threshold to consider a skill learned
parameters.knowledgeLowerBound = 0.01;

% Fundamental complexity (AKA max number of parameters.episodes to learn any skill)
parameters.fundamentalComplexity = 100;
% Trial parameters.episodes
% parameters.episodes  = 0:0.01:parameters.fundamentalComplexity; 
parameters.episodes  = 0:0.1:parameters.fundamentalComplexity; 

% Total number of skills
parameters.totalSkills = 8*64;

% Number of clusters
parameters.totalSkillClusters = 4;

% Number of skills per cluster
parameters.skillsPerCluster = parameters.totalSkills/parameters.totalSkillClusters;

% Base learning rate
parameters.alpha_min = -log(parameters.knowledgeLowerBound)/parameters.fundamentalComplexity;%0.05;
parameters.alpha_max = 1.5*parameters.alpha_min;

parameters.delta = -log(parameters.knowledgeLowerBound)/parameters.skillsPerCluster;%0.05;

parameters.eta_0   = -0*0.1;
parameters.eta_std = 0.1;%0.3*parameters.eta_0;

parameters.gamma_0   = -0*4*0.05;
parameters.gamma_std = 0.5*4*0.05;%parameters.gamma_0;

f = @(eta, N_zeta) eta.*N_zeta+1;
g = @(delta, N_zeta) exp(-delta*N_zeta);

% number of robots available
parameters.maxNumberOfRobots = 128;
parameters.totalSimulationScenarios = 5;


parameters.cl_distributed = 0;
parameters.enableSharing = 0;

%% **************************************************************************
% Dynamics of ISOLATED LEARNING 
% *************************************************************************
clc
close all
rng('default')
clear c_jk_iso_parameters.episodes

parameters.enableSharing = 0;
parameters.enableSharing = 0;
for scenario = 1:parameters.totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]
        fig       = figure('color','w');
        % Transfer learning factor
        beta_k   = 0;
        c_jk_iso = zeros(parameters.skillsPerCluster, parameters.totalSkillClusters);
        % Loop over clusters
        for skillCluster = 1:parameters.totalSkillClusters
            % disp(['Running Scenario: Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            formatSpec = 'Scenario: %4i | Cluster: %4i | No. Robots: %4i\n';
            fprintf(formatSpec,scenario, skillCluster, numberOfRobots)
            subplot(1,parameters.totalSkillClusters,skillCluster)
            % Loop over skills
            for learnedSkills = 0:(parameters.skillsPerCluster/numberOfRobots)-1
                j          = learnedSkills+1;
                initialRemainingKnowedge = ones(numberOfRobots,1);
                % a          = -parameters.alpha_min;
                % F          = a*eye(numberOfRobots);
                a          = -(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));
                F          = diag(a);
                remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, parameters.episodes, initialRemainingKnowedge));
                loglog(parameters.episodes,remainingKnowledge,'k-','LineWidth',1)
                hold on
                % c_jk_iso(j, k) = -log(epsilon)/parameters.alpha_min;
                c_jk_iso(j, skillCluster) = ceil(mean(parameters.episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:numberOfRobots,'UniformOutput',false)))));
            end
            p = area(parameters.episodes, parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(skillCluster),'})$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{IsL})}_{j,',num2str(skillCluster),'}$'],'FontSize',25,'Interpreter','latex')
            xlim([0 parameters.episodes(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
        end
        % legend(skill_labels)
        for ax=1:parameters.totalSkillClusters
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
%% **************************************************************************
% Dynamics of INCREMENTAL LEARNING 
% *************************************************************************
clc
close all

clear c_jk_il_parameters.episodes

parameters.enableSharing = 0;
parameters.cl_distributed = 0;
for scenario = 1:parameters.totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]
        fig     = figure('color','w');
        % Self learning factor
        Alpha  = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1)); 
        % Transfer learning factor
        beta_k  = 0;
        c_jk_il = zeros(parameters.skillsPerCluster,parameters.totalSkillClusters);
        % Loop over clusters
        for skillCluster = 1:parameters.totalSkillClusters
            %disp(['Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            formatSpec = 'Scenario: %4i | Cluster: %4i | No. Robots: %4i\n';
            fprintf(formatSpec,scenario, skillCluster, numberOfRobots)
            subplot(1,parameters.totalSkillClusters,skillCluster)
            % Loop over skills
            for learnedSkills = 0:(parameters.skillsPerCluster/numberOfRobots)-1
                j          = learnedSkills+1;
                initialRemainingKnowedge = (1-beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                % a          = -parameters.alpha_min*((1-beta_k)^(-1))*f(parameters.eta_0, learnedSkills);
                % F          = a*eye(numberOfRobots);
                eta_robots = abs(0.1*parameters.eta_0.*randn(numberOfRobots,1) + parameters.eta_0);
                F          = -Alpha.*((1-beta_k)^(-1)).*f(eta_robots, learnedSkills);


                remainingKnowledge   = ...
                    transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                    fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        learnedSkills,...
                        sigmaBar, ...
                        numberOfRobots, ...
                        Alpha, ...
                        beta_k, ...
                        parameters), parameters.episodes, initialRemainingKnowedge)); 





                %remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, parameters.episodes, initialRemainingKnowedge));
                
                loglog(parameters.episodes,remainingKnowledge,'k-','LineWidth',1)
                hold on
                %c_jk_il(j, k) = -(log(epsilon) + parameters.delta*learnedSkills)/(parameters.alpha_min*f(parameters.eta_0, learnedSkills));
                c_jk_il(j, skillCluster) = ceil(mean(parameters.episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:numberOfRobots,'UniformOutput',false)))));
            end
            disp(['Skills seen:' num2str(j)]);
            p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(skillCluster),'})$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{IL})}_{j,',num2str(skillCluster),'}$'],'FontSize',25,'Interpreter','latex')
            xlim([0 parameters.episodes(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
        end
        % legend(skill_labels)
        for ax=1:parameters.totalSkillClusters
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
clear c_jk_til_parameters.episodes

parameters.enableSharing = 0;
parameters.cl_distributed = 0;
for scenario = 1:parameters.totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = [2,4,8,16,32,64,128]
        fig       = figure('color','w');
        % Self learning factor
        Alpha  = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));         
        c_jk_itl  = zeros(parameters.skillsPerCluster, parameters.totalSkillClusters);
        % Loop over clusters
        for skillCluster = 1:parameters.totalSkillClusters
            % disp(['Cluster: ',num2str(k),' with No. Robots: ', num2str(numberOfRobots), ' --------------------'])
            formatSpec = 'Scenario: %4i | Cluster: %4i | No. Robots: %4i\n';
            fprintf(formatSpec,scenario, skillCluster, numberOfRobots)        
            subplot(1,parameters.totalSkillClusters,skillCluster)
            % Transfer learning factor
            % beta_k = (k-1)/parameters.totalSkillClusters;
            beta_k = (skillCluster-1)/parameters.totalSkillClusters.* rand(numberOfRobots,1);
            % Loop over skills
            for learnedSkills = 0:(parameters.skillsPerCluster/numberOfRobots)-1
                j          = learnedSkills+1;
                initialRemainingKnowedge = (1 - beta_k).*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                % a          = -parameters.alpha_min*((1 - beta_k)^(-1))*f(parameters.eta_0, learnedSkills);
                % F          = a*eye(numberOfRobots);
                % eta_robots = abs(0.1*parameters.eta_0.*randn(numberOfRobots,1) + parameters.eta_0);
                % F          = -Alpha.*((1-beta_k).^(-1)).*f(eta_robots, learnedSkills);
                % remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, parameters.episodes, initialRemainingKnowedge));    
                
                
                remainingKnowledge   = ...
                    transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                    fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        learnedSkills,...
                        sigmaBar, ...
                        numberOfRobots, ...
                        Alpha, ...
                        beta_k, ...
                        parameters), parameters.episodes, initialRemainingKnowedge));                 
                
                
                loglog(parameters.episodes,remainingKnowledge,'k-','LineWidth',1)
                ylim([-0.1 1.1])
                hold on
                %c_jk_itl(j, k) = -(log(epsilon*(1 - beta_k).^(-1)) + parameters.delta*learnedSkills)/(parameters.alpha_min*(1 - beta_k)^(-1)*f(parameters.eta_0, learnedSkills));
                c_jk_itl(j, skillCluster) = ceil(mean(parameters.episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:numberOfRobots,'UniformOutput',false)))));
            end
            p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(skillCluster),'})$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{TIL})}_{j,',num2str(skillCluster),'}$'],'FontSize',25,'Interpreter','latex')
            xlim([0 parameters.episodes(end)])
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

%%
clc
close all
X = linspace(-0.1,0.1,1000);

% Evaluate the pdf for the distribution at the points in X, and then plot the result.

F = pearspdf(X,0.05,0.05*0.5,-1.5,10);
plot(X,F)
hold on
xlim([-0.1,0.1])

%%
clc
% * NOTE: A Pearson distribution with a skewness of 0 and kurtosis of 3 is equivalent to the normal distribution.
% parameters.eta_0 = 0.2;
% parameters.eta_std = 0.3;
close all
X = linspace(-0.1,0.5,1000);
% F = pearspdf(X, parameters.eta_0, parameters.eta_std*parameters.eta_0, -4, 100);
% for kurt =3:20
F = pearspdf(X, parameters.eta_0, parameters.eta_std, 1, 10);
figure('Color','w')
plot(X,F)
hold on
plot(parameters.eta_0*ones(10,1),linspace(0,max(F),10),'k--')
pause(1)
% end
xlim([-0.1,0.5])
box on
%%
pearsrnd(0.05, 0.05*0.5, -1.5, 10, 5,1)
%%
clc
for i=1:1000
    out = pearsrnd(parameters.eta_0, parameters.eta_std, 1, 10,5,1)
    if any(out<0)
        disp("HERE");
    end
end
%% **************************************************************************
% Dynamics of COLLECTIVE LEARNING (ALTERNATIVE)
% *************************************************************************
% * NOTE: all m robots placed in one cluster at a time concurrently 
%         exchanging knowledge
clc
close all

warning("<<ALL>> m robots placed in one cluster at a time concurrently exchanging knowledge")
pause(2)
clear c_jk_cl_parameters.episodes

tic

parameters.enableSharing  = 1;
parameters.cl_distributed = 0;
for scenario = 1%1:parameters.totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    for numberOfRobots = 4%[2,4,8,16,32,64,128]   
        fig     = figure('color','w');

        % Connectivity (adjacency) matrix
        A_full  = ones(numberOfRobots) - eye(numberOfRobots);
        A       = A_full;
                
        c_jk_cl = zeros(parameters.skillsPerCluster, parameters.totalSkillClusters);
        % Loop over clusters
        for skillCluster = 1:parameters.totalSkillClusters
            subplot(1,parameters.totalSkillClusters,skillCluster)
            fprintf('Scenario: %4i | Skill Cluster: %4i | No. Robots: %4i ----\n',scenario, skillCluster, numberOfRobots)        
            
            % Set the INTER-cluster knowledge transferability factor
            beta_k = (skillCluster-1)/parameters.totalSkillClusters.*ones(numberOfRobots,1);
            % Loop over skills
            skillsPool =0;
            %for learnedSkills = 0:parameters.skillsPerCluster/numberOfRobots-1    
            for skillsBatch = 1:parameters.skillsPerCluster/numberOfRobots
                % New set of skills -> new learning rate
                Alpha   = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));
                % Keep track of the number of learned skills
                if (skillsBatch==1)
                    learnedSkills = 0;
                else
                    learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
                end
                j          = learnedSkills + 1;
                initialRemainingKnowedge = (1 - beta_k).*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                
                remainingKnowledge   = ...
                    transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                    fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        learnedSkills,...
                        sigmaBar, ...
                        numberOfRobots, ...
                        Alpha, ...
                        beta_k, ...
                        parameters), parameters.episodes, initialRemainingKnowedge));                
           
                % Get the compplexity
                c_jk_cl(j, skillCluster) = ceil(mean(parameters.episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:numberOfRobots,'UniformOutput',false)))));
                % loglog(parameters.episodes,mean(remainingKnowledge,1),'k-','LineWidth',1)
                loglog(parameters.episodes,remainingKnowledge,'k-','LineWidth',1,'Color',rand(1,3))
                % stairs(parameters.episodes,remainingKnowledge','k-','LineWidth',1,'Color',rand(1,3))
                if any(remainingKnowledge(:,end)==1)
                    warning('There are unlearned skills')
                end
                % plot(c_jk_cl(j, k),max(mean(remainingKnowledge,1)),'ko','LineWidth',1)
                % plot(parameters.episodes,remainingKnowledge,'LineWidth',2)
                xlim([0 parameters.episodes(end)])
                % ylim([1E-3 1E0])
                ylim([0.008 1E0])
                % xticks([0 50 100])
                xticks([10^0 10^1 10^2])
                xticklabels({'1','10', '$c_0$'})
                yticks([1E-3 1E-2 1E-1 1E0])
                yticklabels({'','$\epsilon$', '', '1'})
                set(gca,'TickLabelInterpreter','latex')
                hold on
            end
            p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
            title(['$(r_i,\mathcal{Z}_{',num2str(skillCluster),'}$)'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CL})}_{j,',num2str(skillCluster),'}$'],'FontSize',25,'Interpreter','latex')
        end
        for ax=1:parameters.totalSkillClusters
            fcn_scrpt_prepare_graph_science_std([], fig.Children(ax), [], [], [], 18, 3, 0.25)
        end
        fcn_scrpt_prepare_graph_science_std(fig, [], [], [], [], 18, 3, 0.25)
        pause(2)
        tightfig(fig);
        
    
        SAVE_FIG = 0;
        if numberOfRobots == 2 && SAVE_FIG == 1
            exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_sequential_collective_learning.png'),'Resolution',600)
            close(gcf);
        else
            % close(fig)
        end        
    
        c_jk_cl_seq_episodes(robotBatchIndex,:,scenario) = [sum(c_jk_cl,1), sum(c_jk_cl,'all')] ;
        robotBatchIndex                     = robotBatchIndex + 1; 
    end
end
simTime = toc;

%% ========================================================================
%  ALTERNATIVE robot distribution: EQUAL robots per cluster

close all
clc

warning("<<EQUAL>> number of robots per cluster")
pause(2)
clear c_jk_cl_dist_parameters.episodes

parameters.enableSharing  = 1;
parameters.cl_distributed = 1;
for scenario = 1:parameters.totalSimulationScenarios
    robotBatchIndex = 1;
    % Loop over robots
    % * NOTE: index starts in 4 to have at least one robot per cluster
    for numberOfRobots = [4,8,16,32,64,128]          
        fig = figure('color','w');
        c_jk_cl_dist  = zeros(parameters.totalSkills/numberOfRobots, 1);%zeros(N_Z, N_K);
        
        % New set of skills, new learning rates
        Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));

        for learnedSkills = 0:(parameters.totalSkills/numberOfRobots)-1
            % Beta adapts at every cycle
            % NOTE: From each cluster, only a fraction of knowlede can be transfered
            beta_k = numberOfRobots*learnedSkills/parameters.totalSkills; 
            j        = learnedSkills + 1;
            fprintf('Scenario: %1i |Robots: %3i | Skills batch: %2i | Cluster knowledge: %1.3f \n',scenario,numberOfRobots, j,beta_k)
            initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, numberOfRobots*learnedSkills).*ones(numberOfRobots,1);
            
            % Run remaining knowledge dynamics
            remainingKnowledge = transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                        fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        numberOfRobots*learnedSkills,...
                        sigmaBar, ...
                        numberOfRobots, ...
                        Alpha, ...
                        beta_k, ...
                        parameters), parameters.episodes, initialRemainingKnowedge)); 
    
            c_jk_cl_dist(j) = ...
                ceil(mean(parameters.episodes(cell2mat(arrayfun(@(i) min(find(remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:numberOfRobots,'UniformOutput',false)))));
            
            loglog(parameters.episodes,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
    
            xlim([0 parameters.episodes(end)])
            ylim([1E-3 1E0])
            xticks([0 50 100])
            xticklabels({'0','', '$c_0$'})
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
            hold on
        end
    
        p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
        plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
        % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
        xlabel('Episodes','FontSize',25)
        ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CLd})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
        fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 18/2, 3, 1)
        pause(1)
        tightfig(fig);
        c_jk_cl_dist_episodes(robotBatchIndex, scenario) = sum(c_jk_cl_dist);
        robotBatchIndex                        = robotBatchIndex + 1; 
    end
end

%% Total number of parameters.episodes
complexities = {c_jk_cl_seq_episodes, c_jk_cl_dist_episodes};

% complexities = {c_jk_iso_parameters.episodes,c_jk_il_parameters.episodes,c_jk_til_parameters.episodes,c_jk_cl_parameters.episodes,c_jk_cl_dist_parameters.episodes};

cmap = colororder();
clc
close all
fig = figure('color','w');
p = NaN(1,numel(complexities));
hold on
for index = 1:numel(complexities)
    if index<numel(complexities)
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
% plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
xlabel('Number of robots','FontSize',25)
ylabel('Complexity (episodes per skill)','FontSize',25)
leg = legend(p,'CL$_\mathrm{sequential}$','CL$_\mathrm{distributed}$');
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
        'figures','total_parameters.episodes_per_n_robots_v1.png'),'Resolution',600)
    close(gcf);
end  
% set(gca, 'YScale', 'log')

%% Total number of parameters.episodes
complexities = {c_jk_iso_episodes, ...
                c_jk_il_episodes, ...
                c_jk_til_parameters, ...
                c_jk_cl_seq_episodes, ...
                c_jk_cl_dist_episodes};

cmap = colororder();
clc
close all
fig = figure('color','w');
p = NaN(1,numel(complexities));
hold on
for index = 1:numel(complexities)
    if index<numel(complexities)
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
ylabel('Complexity [episodes for all skills]','FontSize',25)
leg = legend(p,'IsL','IL','TIL','CLs','CLd');
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

%% Total number of parameters.episodes
close all
clc

c_jk_cl_parameters.episodes_mean = c_jk_cl_parameters.episodes(:,end,3);
c_jk_cl_parameters.episodes_std = ceil(std(c_jk_cl_parameters.episodes,[],3));
x = 1:7;
uppeBound = c_jk_cl_parameters.episodes_mean(:,end) + c_jk_cl_parameters.episodes_std(:,end);
lowerBound = c_jk_cl_parameters.episodes_mean(:,end) - c_jk_cl_parameters.episodes_std(:,end);
x2 = [x, fliplr(x)];
inBetween = [uppeBound', fliplr(lowerBound)'];
fill(x2, inBetween, 'g');
hold on;

%%
close all
patch([x fliplr(x)], [lowerBound'  fliplr(uppeBound')], [0.6  0.7  0.8],'FaceColor',[0.5 0.5 0.5],'FaceAlpha',0.25,'EdgeColor','w');
hold on
plot(x, c_jk_cl_parameters.episodes_mean(:,end), 'k', 'LineWidth', 1);
% set(gca, 'YScale', 'log')


%%

% factor = [2,4,8,16,32,64,128]'*105E3;
fig = figure('color','w');
% p0 = semilogy([2,4,8,16,32,64,128],c_jk_iso_parameters.episodes(:,end),'o-');
% hold on
% p1 = plot([2, 4, 8, 16, 32, 64, 128],c_jk_il_parameters.episodes(:,end),'o-');
% p2 = plot([2, 4, 8, 16, 32, 64, 128],c_jk_til_parameters.episodes(:,end),'o-');
% p3 = plot([2, 4, 8, 16, 32, 64, 128],c_jk_cl_parameters.episodes(:,end),'o-');
% p4 = plot(   [4, 8, 16, 32, 64, 128],c_jk_cl_dist_parameters.episodes,'o-');
p0 = semilogy(1:7,ceil(c_jk_iso_parameters.episodes(:,end)),'o-');
hold on
p1 = plot(1:7,ceil(c_jk_il_parameters.episodes(:,end)),'o-');
p2 = plot(1:7,ceil(c_jk_til_parameters.episodes(:,end)),'o-');
p3 = plot(1:7,c_jk_cl_parameters.episodes(:,end),'o-');
p4 = plot(2:7,c_jk_cl_dist_parameters.episodes,'o-');

plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)
% xticks([2,4,8,16,32,64,128])
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
% xlabel('m','FontSize',25)
% ylabel('$C_\mathcal{S}$','FontSize',25,'Interpreter','latex')
xlabel('Number of robots','FontSize',25)
ylabel('Complexity [parameters.episodes]','FontSize',25)
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
        'figures','total_parameters.episodes_per_n_robots.png'),'Resolution',600)
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
p0 = semilogy([2,4,8,16,32,64,128],factor.*c_jk_iso_parameters.episodes(:,end),'o-');
hold on
p1 = plot([2, 4, 8, 16, 32, 64, 128],factor.*c_jk_il_parameters.episodes(:,end),'o-');
p2 = plot([2, 4, 8, 16, 32, 64, 128],factor.*c_jk_til_parameters.episodes(:,end),'o-');
p3 = plot([2, 4, 8, 16, 32, 64, 128],factor.*c_jk_cl_parameters.episodes(:,end),'o-');
p4 = plot(   [4, 8, 16, 32, 64, 128],factor(2:end).*c_jk_cl_dist_parameters.episodes','o-');
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
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','total_parameters.episodes_per_n_robots'));
disp('Printing done!')





%% Alternative robot distribution for collective learning

% Robots concurrently exchanging knowledge in one cluster and moving to the 


clc
close all
fig = figure('color','w');


r        = numberOfRobots;
c_jk_cl  = zeros(parameters.skillsPerCluster/numberOfRobots, parameters.totalSkillClusters);
kappa = zeros(1,parameters.skillsPerCluster+1);
for skillCluster = 1:parameters.totalSkillClusters
    disp(['Cluster ',num2str(skillCluster),' -------------------------'])
    subplot(1,parameters.totalSkillClusters,skillCluster)
    beta_k = (skillCluster-1)/parameters.totalSkillClusters;
    for learnedSkills = 0:parameters.skillsPerCluster/numberOfRobots-1    
        j        = learnedSkills+1;
        initialRemainingKnowedge = (1- beta_k)*g(parameters.delta, r*learnedSkills).*ones(numberOfRobots,1);
        a        = -parameters.alpha_max*f(parameters.eta_0, r*learnedSkills)*(1- beta_k)^(-1);
        gamma_0    = 0.005;%-0.3*(-(a)/max(eig(abs(A))));
        F        = a*eye(r) + gamma_0*A;
        lambda_F = eig(F)
        remainingKnowledge   = transpose(ode4(@(n,sigmaBar) F*sigmaBar, parameters.episodes, initialRemainingKnowedge));
%         for i=0:m-1
%             c_jk_cl(j+i, k) = n(min(find(sigmaBar(i+1,:)<epsilon))); 
%         end
%         for r_i=1:r
%             c_jk_cl(r_i, k) = n(min(find(sigmaBar(r_i,:)<epsilon)));
%         end
        c_jk_cl(j, skillCluster) = parameters.episodes(min(find(remainingKnowledge(1,:)<parameters.knowledgeLowerBound)));
        semilogx(parameters.episodes,remainingKnowledge,'LineWidth',3)
        hold on
%         kappa(N_zeta + 1)  = sum(sigmaBar);
    end
    p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(skillCluster),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(skillCluster),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 parameters.episodes(end)])
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
for skillCluster = 1:parameters.totalSkillClusters
    A = G*A_i;
    disp('A:');
    disp(A)
    disp('Eigenvalues:')
    disp([(eig(A)),(eig(A_i)),abs(eig(A))-abs(eig(A_i))]) 
    
    
    disp('Det:')    
    disp(det(A))
%     close all
    learning_type_1 = 'collective_sat';
    % (t, x, A, alpha, s, parameters.fundamentalComplexity, rate, threshold, type)
    if cycle ==1
        var = 1;
      
    else
        var = (cycle-1)*numberOfRobots;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, parameters.alpha_max, s, parameters.fundamentalComplexity, tau, parameters.knowledgeLowerBound, learning_type_1), ...
                                             parameters.episodes, ...
                                             1-(cycle-1)/(N/numberOfRobots)*ones(numberOfRobots,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(parameters.episodes,x,'LineWidth',2);
    hold on
end
plot(parameters.episodes,exp(A_i(1,1)*parameters.episodes),'k--','LineWidth',2)
p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 parameters.episodes(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')


%% **************************************************************************
% Dynamics of colletive learning 
% *************************************************************************
clc
close all
clear A

% This variable determines when knowlege collection is completed
parameters.knowledgeLowerBound = 0.01;

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
parameters.alpha_max     = -log(parameters.knowledgeLowerBound)/N;  % log(0.01)/N;%(rate/N);

% Initial values of the subsequent skills based on alpha
initialRemainingKnowedge  = exp(-parameters.alpha_max*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:parameters.fundamentalComplexity*N;
parameters.episodes  = 0:0.01:100; 

% Initial complexity
parameters.fundamentalComplexity        = 100;

% Initial knowledge aquisition rate per episode
% *NOTE: determined to satisfy <threshold = exp(-(rate/parameters.fundamentalComplexity)*parameters.fundamentalComplexity)>
tau       = -log(parameters.knowledgeLowerBound)*ones(numberOfRobots,1);
A_i       = -diag(tau/parameters.fundamentalComplexity);

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


X = NaN(numberOfRobots,numel(parameters.episodes),N/numberOfRobots);
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
    % (t, x, A, alpha, s, parameters.fundamentalComplexity, rate, threshold, type)
    if cycle ==1
        var = 1;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, parameters.alpha_max, s, parameters.fundamentalComplexity, tau, parameters.knowledgeLowerBound, learning_type_1), ...
                                             parameters.episodes, ...
                                             ones(numberOfRobots,1));        
    else
        var = (cycle-1)*numberOfRobots;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, parameters.alpha_max, s, parameters.fundamentalComplexity, tau, parameters.knowledgeLowerBound, learning_type_1), ...
                                             parameters.episodes, ...
                                             1-(cycle-1)/(N/numberOfRobots)*ones(numberOfRobots,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(parameters.episodes,x,'LineWidth',2);
    hold on
end
plot(parameters.episodes,exp(A_i(1,1)*parameters.episodes),'k--','LineWidth',2)
p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 parameters.episodes(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')

%%
% -------------------------------------------------------------------------
learning_type_2 = 'collective';
% (t, x, A, alpha, s, parameters.fundamentalComplexity, rate, threshold, type)
x_nonebounded = ode4(@(t,x) ...
      fcn_general_dynamics(t, x, A, parameters.alpha_max, s, parameters.fundamentalComplexity, tau, parameters.knowledgeLowerBound, learning_type_2), ...
                                     parameters.episodes, ...
...
ones(N,1));

figure('Color','w')
p = plot(parameters.episodes,x_nonebounded,':','LineWidth',1.5);
hold on
set(gca,'ColorOrderIndex',1)
p = plot(parameters.episodes,x,'LineWidth',2);
plot(parameters.episodes,exp(A_i(1,1)*parameters.episodes),'k--','LineWidth',2)
p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 parameters.episodes(end)])
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
parameters.alpha_max     = -log(parameters.knowledgeLowerBound)/N;%-log(0.01)/N;%(rate/N);
% alpha     = (rate/N);
initialRemainingKnowedge  = exp(-parameters.alpha_max*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:parameters.fundamentalComplexity*N;
parameters.episodes  = 0:0.01:1000; 
learning_type = 'incremental_3';
x = ode4(@(t,x) fcn_general_dynamics(t, x, A, parameters.alpha_max, s, parameters.fundamentalComplexity, tau, parameters.knowledgeLowerBound,...
                learning_type), ...
                                     parameters.episodes, ...
                                     zeros(N,1) +1*initialRemainingKnowedge(1:end-1)');
x = x';
figure('Color','w')
if strcmp(learning_type,'incremental_2')
%     p = plot(n,initialRemainingKnowedge(1:end-1)'-x,'LineWidth',2);
    p = plot(parameters.episodes,x,'LineWidth',2);
else
    p = plot(parameters.episodes, x,'LineWidth',2);
end
hold on

% p = plot(n,(1-initialRemainingKnowedge(1:end-1))' - x,'LineWidth',2);
% p = plot(n, - x,'LineWidth',2);
% hold on
plot(parameters.episodes(end),initialRemainingKnowedge(1:end-1),'*')
xlabel('Episodes [n]','FontSize',25)
ylabel('Knowledge ','FontSize',25)
% xlim([0 sum(parameters.fundamentalComplexity*initialRemainingKnowedge)])
ylim([0 1])
%
title('Incremental')

% clear indx
% for j = 1:size(x,1)
%     parameters.delta  = x(j,:) - initialRemainingKnowedge(j);
%     indx(j) = find(abs(parameters.delta) < threshold,1,'first');
% %     disp(n(indx(j)))
% end
% figure('Color','w')
% plot(1:N,parameters.fundamentalComplexity*initialRemainingKnowedge(1:end-1),'--','LineWidth',2)
% hold on
% plot(1:N,[n(indx(1)), diff(n(indx(:)))],'LineWidth',2)
% xlabel('Skills','FontSize',25)
% ylabel('Complexity ','FontSize',25)
% legend('Ideal','Actual','FontSize',10)
% xlim([1 N])


%%

sig_bar = @(Nj,alpha) exp(-alpha*Nj);
parameters.episodes = 0:parameters.fundamentalComplexity;
clear indx
clc
close all
figure('Color','w')
% plot(Nj,sig_bar(Nj,(5/N)),'LineWidth',3)
% hold on
semilogy(parameters.episodes,0.01*ones(size(parameters.episodes)),'k:','LineWidth',3) % Threshold for skill completion
hold on
% The next two plots should be identical
% plot(Nj,0.3*sig_bar(Nj,(5/N)))
% stairs(Nj,sig_bar(Nj -log(0.3)/(5/N),(5/N)))
bsig = [];
for i=1:N
    bsig(i,:) = initialRemainingKnowedge(i)*sig_bar(parameters.episodes,(tau/parameters.fundamentalComplexity));
    indx(i) = find(bsig(i,:)<0.01,1,'first');
    disp(parameters.episodes(indx(i)))
    p = plot(parameters.episodes,initialRemainingKnowedge(i)*sig_bar(parameters.episodes,(tau/parameters.fundamentalComplexity)));
    semilogy(parameters.episodes(indx(i))*ones(numel(0:0.00001:1),1),0:0.00001:1,'--','Color',p.Color);
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

x = ode4(@(t,x) -A(1,1)*x,parameters.episodes,1);
figure
stairs(parameters.episodes,x)
xlabel('Episodes [n]','FontSize',25)

%% ************************************************************************

clearvars
close all
clc
clc
parameters.fundamentalComplexity      = 100;
alpha_i = 10;
parameters.totalSkills     = 6;
parameters.totalSkillClusters     = 2;
N       = parameters.totalSkills/parameters.totalSkillClusters;
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
parameters.episodes = 0:0.001:parameters.fundamentalComplexity;
% n=0;
f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
x = f(parameters.episodes);

f_i = @(t,x) -A.*eye(N)*x; 
f_c = @(t,x) -A*x; 
% x = ode4(f,n,sum(A,2));
x_i = ode4(f_i,parameters.episodes,ones(N,1));
x_c  = ode4(f_c,parameters.episodes,ones(N,1));


min(x_c(:))
figure('Color',[1 1 1])
plot(parameters.episodes,x_i,'k','LineWidth',1.5)
hold on
% plot(n,mean(x,2),'k--')
plot(parameters.episodes,x_c)
xlabel('Episodes n','FontSize',25)
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
xlim([0 5])
%%
close all
 plot(parameters.episodes,exp(-3*parameters.episodes) + exp(-8*parameters.episodes))
 plot(parameters.episodes,0.7*exp(-3*parameters.episodes))
 hold on
 plot(parameters.episodes,0.3*exp(-8*parameters.episodes))
 plot(parameters.episodes,0.7*exp(-3*parameters.episodes) + 0.3*exp(-8*parameters.episodes),'k--')
 xlim([0 10])
%% ************************************************************************
% SIMULATION OF THE COMPLETE SYSTEM
% *************************************************************************

clearvars
close all
clc
clc
parameters.fundamentalComplexity      = 100;
alpha_i = 10;
parameters.totalSkills     = 4;
parameters.totalSkillClusters     = 2;
N       = parameters.totalSkills/parameters.totalSkillClusters;

Spp   = sympositivedefinitefactory(N);
A1     = Spp.rand();
A1     = A1 + abs(min(A1,[],'all'));

Spp   = sympositivedefinitefactory(N);
A2     = Spp.rand();
A2     = A2 + abs(min(A2,[],'all'));

B = 0.1/3*ones(N);

A = [A1, B; B, A2]

[V,D] = eig(-A/parameters.fundamentalComplexity)
% c     = (V\ones(N,1));
%
close all
parameters.episodes = 0:0.001:parameters.fundamentalComplexity;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
% f = @(t,x) -eye(size(A))*x; 
% x = ode4(f,n,sum(A,2));
x = ode4(f,parameters.episodes,ones(size(A,1),1));

min(x(:))
figure('Color',[1 1 1])
plot(parameters.episodes,x(:,1:2),'b--')
hold on
plot(parameters.episodes,x(:,3:4),'r')
% plot(n,sum(x,2),'k:')
xlabel('Episodes n','FontSize',25)
xlim([0 5])
ylabel('$\bar{\sigma}^{(C)}_{j,k}$','Interpreter','latex','FontSize',25)

selfDynamicsMatrix = zeros(parameters.totalSkillClusters, 1);
selfDynamicsMatrix(1) = plot(NaN,NaN,'--b');
selfDynamicsMatrix(2) = plot(NaN,NaN,'-r');
leg = legend(selfDynamicsMatrix, 'k_1','k_1');
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
parameters.episodes = 0:0.01:50;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
x = ode4(f,parameters.episodes,sum(W,2));



min(x(:))
plot(parameters.episodes,x)
hold on
plot(parameters.episodes,mean(x,2),'k--')

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
parameters.episodes = 0:0.001:100;
f = @(t,x) -L*x; 
% x = ode4(f,n,sum(A,2));
% x = ode4(f,n,ones(N,1));
x0 = rand(N,1);
x0 = N*(x0/sum(x0));
x = ode4(f,parameters.episodes,x0);
plot(parameters.episodes,x)
xlabel('Episodes [n]','FontSize',25)
% xlim([0 10])
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
