function [c_jk_cl_dist_episodes, learnedSkillsStorage, clusterKnowledge] = ...
    simulateDistributedCollectiveKnowledgeDynamics(NameValueArgs)

    arguments
        NameValueArgs.eta_0
        NameValueArgs.gamma_0
        NameValueArgs.parameters
    end


    eta_0      = NameValueArgs.eta_0;
    gamma_0    = NameValueArgs.gamma_0;
    parameters = NameValueArgs.parameters;
    
    g = @(delta, N_zeta) exp(-delta*N_zeta);    
    
    % Overwrite default eta and gamma parameters
    parameters.eta_0   = eta_0;
    parameters.gamma_0 = gamma_0;

    if mod(parameters.totalSkills,parameters.numberOfRobots)==0
        numberOfSkillBatches      = parameters.totalSkills/parameters.numberOfRobots;
        UNEVEN_ROBOT_DISTRIBUTION = 0;
    else
        numberOfSkillBatches      = floor(parameters.totalSkills/parameters.numberOfRobots) + 1;
        UNEVEN_ROBOT_DISTRIBUTION = 1;
    end
    
    % NOTE: Subindex jk means SKILL j in CLUSTER k
    complexity_jk_CL_Distributed = zeros(numberOfSkillBatches, 1);
    clusterKnowledge    = zeros(numberOfSkillBatches, 1);

% =========================================================================
fig = figure('color','w');
p = area(parameters.episodes, parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
hold on
plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
% set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')
% title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
xlabel('Episodes','FontSize',25)
ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{DCL})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
% Generate a colormap (e.g., 'parula', 'jet', 'hot', 'cool', etc.)
cmap = colormap('lines');

% Create indices to pick colors evenly from the colormap
colorIndices = linspace(1, size(cmap, 1), numberOfSkillBatches);

% Interpolate the colormap to get the desired number of colors
selectedColors = interp1(1:size(cmap, 1), cmap, colorIndices);
% =========================================================================    
    
    % New set of skills => new learning rates
    Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(parameters.numberOfRobots,1));
    numberOfLearnedSkills = 0;
% Alpha = mean([parameters.alpha_min,parameters.alpha_max])*eye(parameters.numberOfRobots);
% learnedSkills = 0;
    for skillsBatch = 1:numberOfSkillBatches
        % NOTE: the transferrableKnowledgeFraction (i.e., beta_k) varies at
        %       every cycle. From each cluster, only a fraction of knowlede
        %       can be transfered.
        % beta_k = parameters.numberOfRobots*learnedSkills/parameters.totalSkills;
        
        totalTransferrableKnowledgeFraction = (numberOfLearnedSkills/parameters.totalSkills)*parameters.enableTransfer;


        clusterKnowledge(skillsBatch)  = totalTransferrableKnowledgeFraction;
        j      = skillsBatch;%learnedSkills + 1;
        fprintf('Gamma scenario: %2i | Eta scenario: %2i | Skills batch: %2i/%3i | Cluster knowledge: %1.3f \n',1, 1, skillsBatch, numberOfSkillBatches, totalTransferrableKnowledgeFraction)
        % initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, parameters.numberOfRobots*learnedSkills).*ones(parameters.numberOfRobots,1);
        
        if UNEVEN_ROBOT_DISTRIBUTION == 0
            initialRemainingKnowledge = (1 - totalTransferrableKnowledgeFraction)*g(parameters.delta, numberOfLearnedSkills).*ones(parameters.numberOfRobots,1);
            remainingKnowledge = transpose( ...
                ode4_sat(@(n,sigmaBar) ...
                    fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        numberOfLearnedSkills,...
                        sigmaBar, ...
                        parameters.numberOfRobots, ...
                        Alpha, ...
                        totalTransferrableKnowledgeFraction, ...
                        parameters, n), parameters.episodes, initialRemainingKnowledge));       

        elseif (UNEVEN_ROBOT_DISTRIBUTION == 1) 
            if (skillsBatch < numberOfSkillBatches)
                initialRemainingKnowledge = (1 - totalTransferrableKnowledgeFraction)*g(parameters.delta, numberOfLearnedSkills).*ones(parameters.numberOfRobots,1);
                remainingKnowledge = transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                        fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                            numberOfLearnedSkills,...
                            sigmaBar, ...
                            parameters.numberOfRobots, ...
                            Alpha, ...
                            totalTransferrableKnowledgeFraction, ...
                            parameters, n), parameters.episodes, initialRemainingKnowledge));       

            elseif (skillsBatch == numberOfSkillBatches)
                initialRemainingKnowledge = (1 - totalTransferrableKnowledgeFraction)*g(parameters.delta, numberOfLearnedSkills).*ones(parameters.totalSkills-parameters.numberOfRobots*(numberOfSkillBatches-1),1);
                remainingKnowledge = transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                        fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                            numberOfLearnedSkills,...
                            sigmaBar, ...
                            parameters.totalSkills-parameters.numberOfRobots*(numberOfSkillBatches-1), ...
                            Alpha(1:parameters.totalSkills-parameters.numberOfRobots*(numberOfSkillBatches-1),1:parameters.totalSkills-parameters.numberOfRobots*(numberOfSkillBatches-1)), ...
                            totalTransferrableKnowledgeFraction, ...
                            parameters, n), parameters.episodes, initialRemainingKnowledge));  
            end
        end
        % mean(initialRemainingKnowledge)
        % Add the learned skills in the batch to the general pool of
        % learned skills
        numberOfLearnedSkills = numberOfLearnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
        if any(remainingKnowledge(:,end)>parameters.knowledgeLowerBound)
            warning('Some skills could NOT be learned')
        end                

        complexity_jk_CL_Distributed(j) = ...
            ceil( ...
                mean( ...
                    parameters.episodes( ...
                        cell2mat( ...
                            arrayfun( ...
                                @(i) min( ...
                                    find( ...
                                        remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:size(remainingKnowledge,1),'UniformOutput',false)))));
        
        if isnan(complexity_jk_CL_Distributed(j))
            complexity_jk_CL_Distributed(j) = parameters.fundamentalComplexity;
        end

        % loglog(parameters.episodes,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
        % loglog(parameters.episodes,(remainingKnowledge),'LineStyle','-','color',[0.7 0.7 0.7],'LineWidth',1)
        % xlim([parameters.episodes(1), parameters.episodes(end)])
        % xticks([10^0 10^1 10^2])
        % xticklabels({'1','10', '$c_0$'})


        % gammaValues = 0.022*parameters.episodes - 0.2;
        % % semilogy(gammaValues,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
        % % semilogy(gammaValues,remainingKnowledge,'LineStyle','-','color',rand(1,3),'LineWidth',1)
        % loglog(parameters.episodes,remainingKnowledge,'LineStyle','-','color',selectedColors(skillsBatch,:),'LineWidth',1)
        
        
% =========================================================================        
% Calculate mean and standard deviation across all curves
meanCurve = mean(remainingKnowledge,1);
stdCurve  = std(remainingKnowledge, 0, 1);

% Define the upper and lower bounds of the shaded region
upperBound = meanCurve + stdCurve;
lowerBound = max(parameters.knowledgeLowerBound,meanCurve - stdCurve);

% Create the plot
% Plot the shaded region for standard deviation
if ~all(meanCurve<parameters.knowledgeLowerBound)
    patch([parameters.episodes, fliplr(parameters.episodes)], [upperBound, fliplr(lowerBound)], selectedColors(skillsBatch,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.3);
end
% Plot the mean curve
plot(parameters.episodes, meanCurve, 'b-', 'LineWidth', 0.5,'Color',selectedColors(skillsBatch,:));
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
plot(parameters.fundamentalComplexity*ones(100,1), linspace(parameters.knowledgeLowerBound,1,100),'k--')
% =========================================================================        
        
        
        
        
        
        
        
        
        
        
        
        
        % xlim([parameters.episodes(1) parameters.episodes(end)])
        % % xlim([gammaValues(1) gammaValues(end)])
        % xticks(parameters.episodes([1,11,101,1001,5001]))
        % xticklabels(arrayfun(@(i) num2str(gammaValues(i)),[1,11,101,1001,5001],'UniformOutput',false))        
        
        % Format y-axis
        ylim([0.008 1E0])
        yticks([1E-3 1E-2 1E-1 1E0])
        yticklabels({'','$\epsilon$', '', '1'})
        set(gca,'TickLabelInterpreter','latex')
        hold on
        % exportgraphics(gcf,'always_stable.gif','Append',true,'Resolution',300)
        pause(1)
    end




    % p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    % plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
    % % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    % xlabel('Episodes','FontSize',25)
    % ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CLd})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
    title(gca,['$m =',num2str(parameters.numberOfRobots),'~|~\bar{\eta}=',num2str(eta_0),'~|~\bar{\gamma} =', num2str(gamma_0),'$'],'Interpreter','latex','FontSize',11)
    fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 6, 1, 1)

% % =========================================================================
% % Create a second axes object on top of the first
% ax1 = gca;
% ax2 = axes('Position', ax1.Position, ...
%            'XAxisLocation', 'top', ...
%            'YAxisLocation', 'right', ...
%            'Color', 'none', ...
%            'XColor', 'r', ... % Color of the top x-axis
%            'YColor', 'none'); % Hide the right y-axis
% 
% 
%         xlim(ax2, [0 parameters.episodes(end)])
%         xticks(ax2, [10^-1 10^1 10^2 5*10^2])
%         xticklabels(ax2, {'-0.2','0.02','2','10.8'})
% 
% 
% xscale(ax2, 'log')
% 
% linkaxes([ax1, ax2], 'x');
%         % xticklabels(ax2, {'-0.2','','2','','','','','','','','10.4'})
% 
% % Set the x-axis limits and ticks for the second x-axis
% 
% % ax2.Xticks([-0.2,1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
% % ax2.XLim = [-0.2 10.4];
% 
% 
% % ax2.XTick = [1 10 100 1000 2000 3000 4000 5000];
% % ax2.XTick = [10^-1 10^0 10^1 1000 2000 3000 4000 5000];
% % ax2.XTickLabel = {'-0.2','-0.18','-0.10','1.99','4.09','6.19','8.29','10.39'};
% % ax2.XLim = [-0.2 10.4];
% xlabel(ax2, '$\bar{\gamma}$','FontSize',25,Interpreter='latex')
%         % xticklabels({'1','10', '$c_0$'})
% % =========================================================================


    pause(1)
    tightfig(fig);
    c_jk_cl_dist_episodes = (complexity_jk_CL_Distributed);
    learnedSkillsStorage  = numberOfLearnedSkills;
end

% function [c_jk_cl_dist_episodes, learnedSkillsStorage, clusterKnowledge] = simulateDistributedCollectiveKnowledgeDynamics(eta_0, gamma_0, parameters, numberOfRobots)
%     g = @(delta, N_zeta) exp(-delta*N_zeta);    
% 
%     parameters.eta_0   = eta_0;
%     parameters.gamma_0   = gamma_0;
%     fig = figure('color','w');
%     % Subindex jk means skill j cluster k
%     % c_jk_cl_dist  = zeros(parameters.totalSkills/numberOfRobots, 1);
%     if mod(parameters.totalSkills,numberOfRobots)==0
%         numberOfSkillBatches = parameters.totalSkills/numberOfRobots;
%         FLAG = 0;
%     else
%         numberOfSkillBatches = floor(parameters.totalSkills/numberOfRobots) + 1;
%         FLAG = 1;
%     end
%     c_jk_cl_dist  = zeros(numberOfSkillBatches, 1);
%     clusterKnowledge  = zeros(numberOfSkillBatches, 1);
% 
%     % New set of skills, new learning rates
%     Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));
%     learnedSkills = 0;
%     for skillsBatch = 1:numberOfSkillBatches
%         % if (skillsBatch==1)
%         %     learnedSkills = 0;
%         % else
%         %     learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
%         % end                
%         % Beta adapts at every cycle
%         % NOTE: From each cluster, only a fraction of knowlede can be transfered                
%         % beta_k = numberOfRobots*learnedSkills/parameters.totalSkills;
%         beta_k = learnedSkills/parameters.totalSkills;
%         clusterKnowledge(skillsBatch) = beta_k;
%         j      = skillsBatch;%learnedSkills + 1;
%         fprintf('Gamma scenario: %2i | Eta scenario: %2i | Skills batch: %2i | Cluster knowledge: %1.3f \n',1, 1, skillsBatch, beta_k)
%         % initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, numberOfRobots*learnedSkills).*ones(numberOfRobots,1);
% 
%         if FLAG == 0
%             initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
%             remainingKnowledge = transpose( ...
%                 ode4_sat(@(n,sigmaBar) ...
%                     fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
%                     learnedSkills,...
%                     sigmaBar, ...
%                     numberOfRobots, ...
%                     Alpha, ...
%                     beta_k, ...
%                     parameters), parameters.episodes, initialRemainingKnowedge));       
% 
%         elseif (FLAG == 1) 
%             if (skillsBatch < numberOfSkillBatches)
%                 initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
%                 remainingKnowledge = transpose( ...
%                     ode4_sat(@(n,sigmaBar) ...
%                         fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
%                         learnedSkills,...
%                         sigmaBar, ...
%                         numberOfRobots, ...
%                         Alpha, ...
%                         beta_k, ...
%                         parameters), parameters.episodes, initialRemainingKnowedge));       
% 
%             elseif (skillsBatch == numberOfSkillBatches)
%                 initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1),1);
%                 remainingKnowledge = transpose( ...
%                     ode4_sat(@(n,sigmaBar) ...
%                         fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
%                         learnedSkills,...
%                         sigmaBar, ...
%                         parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1), ...
%                         Alpha(1:parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1),1:parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1)), ...
%                         beta_k, ...
%                         parameters), parameters.episodes, initialRemainingKnowedge));  
%             end
%         end
% 
%         learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
%         if any(remainingKnowledge(:,end)==1)
%             warning('There are unlearned skills')
%         end                
% 
%         c_jk_cl_dist(j) = ...
%             ceil( ...
%                 mean( ...
%                     parameters.episodes( ...
%                         cell2mat( ...
%                             arrayfun( ...
%                                 @(i) min( ...
%                                     find( ...
%                                         remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:size(remainingKnowledge,1),'UniformOutput',false)))));
% 
%         if isnan(c_jk_cl_dist(j))
%             % disp('here')
%             c_jk_cl_dist(j) = parameters.fundamentalComplexity;
%         end
% 
% 
%         loglog(parameters.episodes,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
%         xlim([0 parameters.episodes(end)])
%         ylim([0.008 1E0])
%         xticks([10^0 10^1 10^2])
%         xticklabels({'1','10', '$c_0$'})
%         yticks([1E-3 1E-2 1E-1 1E0])
%         yticklabels({'','$\epsilon$', '', '1'})
%         set(gca,'TickLabelInterpreter','latex')
%         hold on
%     end
% 
%     p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
%     plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
%     % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
%     xlabel('Episodes','FontSize',25)
%     ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CLd})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
%     fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 18/2, 3, 1)
%     pause(1)
%     tightfig(fig);
%     c_jk_cl_dist_episodes = sum(c_jk_cl_dist);
%     learnedSkillsStorage  = learnedSkills;
% 
% end