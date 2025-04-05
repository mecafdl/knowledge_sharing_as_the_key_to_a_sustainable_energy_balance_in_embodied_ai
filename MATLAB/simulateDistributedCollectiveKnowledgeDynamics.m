function [c_jk_cl_dist_episodes, learnedSkillsStorage, clusterKnowledge, results] = ...
    simulateDistributedCollectiveKnowledgeDynamics(NameValueArgs)

    arguments
        NameValueArgs.eta_0
        NameValueArgs.gamma_0
        NameValueArgs.parameters
        NameValueArgs.maxNumberOfProducts = 1000;
    end

    eta_0               = NameValueArgs.eta_0;
    gamma_0             = NameValueArgs.gamma_0;
    parameters          = NameValueArgs.parameters;
    maxNumberOfProducts = NameValueArgs.maxNumberOfProducts;
    
    % Overwrite default eta and gamma parameters
    parameters.eta_0   = eta_0;
    parameters.gamma_0 = gamma_0;

    % Function handle for intiial remaining knowledge
    g = @(delta, N_zeta) exp(-delta*N_zeta);    
    

    if parameters.repeatingSkillsFlag == 0
        if mod(parameters.totalSkills,parameters.numberOfRobots)==0
            numberOfSkillBatches      = parameters.totalSkills/parameters.numberOfRobots;
            % Set a flag
            UNEVEN_ROBOT_DISTRIBUTION = 0;
        else
            numberOfSkillBatches      = floor(parameters.totalSkills/parameters.numberOfRobots) + 1;
            % Set a flag
            UNEVEN_ROBOT_DISTRIBUTION = 1;
        end
    else
        numberOfSkillBatches      = maxNumberOfProducts;
        % Set a flag
        UNEVEN_ROBOT_DISTRIBUTION = 0;        
        
    end


    % Create figure to track progress =====================================
    fig = figure('color','w');
    p   = area(parameters.episodes, parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    hold on
    plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
    % set(gca, 'YScale', 'log')
    % set(gca, 'XScale', 'log')
    % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('Episodes','FontSize',25)
    ylabel('$\bar{\boldmath{\sigma}}^{(\mathrm{DCL})}_{j,k}$','FontSize',25,'Interpreter','latex')
    % Generate a colormap (e.g., 'parula', 'jet', 'hot', 'cool', etc.)
    cmap = colormap('lines');
    % Create indices to pick colors evenly from the colormap
    colorIndices = linspace(1, size(cmap, 1), numberOfSkillBatches);
    % Interpolate the colormap to get the desired number of colors
    selectedColors = interp1(1:size(cmap, 1), cmap, colorIndices);
    % =====================================================================
    

    % Loop over skill batches (products) ==================================

    % NOTE: Subindex jk means SKILL j in CLUSTER k
    complexity_jk_CL_Distributed = zeros(numberOfSkillBatches, 1);
    clusterKnowledge             = zeros(numberOfSkillBatches, 1);
    
    % Learning rates PER agent (embodiment dependent)
    % Alpha = mean([parameters.alpha_min,parameters.alpha_max])*eye(parameters.numberOfRobots);
    Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(parameters.numberOfRobots,1));
    
    % Initialization
    numberOfLearnedSkills = 0;    
    seenSkills            = []; % Pool of learned skills 
    parameters.clusterTransferrableKnowledgeFractionPerAgent  = zeros(1,parameters.numberOfRobots);
    
    clear numberOfNewSkills numberOfSeenSkills  skillsInCluster1 skillsInCluster2  skillsInCluster3 skillsInCluster4
    numberOfNewSkills  = zeros(1,maxNumberOfProducts);
    numberOfSeenSkills = zeros(1,maxNumberOfProducts);
    SkillsInCluster   = zeros(parameters.totalSkillClusters,maxNumberOfProducts);
    % for skillsBatch = 1:numberOfSkillBatches
    for skillsBatch = 1:maxNumberOfProducts

        % NOTE: the transferrableKnowledgeFraction (i.e., beta_k) varies at
        %       every cycle. From each cluster, only a fraction of knowlede
        %       can be transfered.
        % beta_k = parameters.numberOfRobots*learnedSkills/parameters.totalSkills;




        % for k = 1:parameters.totalSkillClusters
        %     SkillsInCluster(k,skillsBatch) = numel(intersect(seenSkills,parameters.skillClusters(:,k)));
        % end


        
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
        
        % Trigger warning if there are unlearned skills
        if any(remainingKnowledge(:,end)>parameters.knowledgeLowerBound)
            warning('Some skills COULD NOT be learned')
        end    

        % mean(initialRemainingKnowledge)
        % Add the learned skills in the batch to the general pool of
        % learned skills
        % numberOfLearnedSkills = numberOfLearnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);


        % =========================================================================
        % productSkills                   = randi(parameters.totalSkills,1,parameters.numberOfRobots);
        productSkills                   = randi(parameters.totalSkills,1,parameters.maxNumberOfSkillsPerProduct);
        if parameters.maxNumberOfSkillsPerProduct<parameters.numberOfRobots
            extraSkills   = productSkills(randi(numel(productSkills),1,parameters.numberOfRobots-parameters.maxNumberOfSkillsPerProduct));
            productSkills = [productSkills, extraSkills];
        end
  
        numberOfNewSkills(skillsBatch)  = numel(unique([seenSkills, productSkills])) - numel(seenSkills);
        seenSkills                      = unique([seenSkills, productSkills]);
        numberOfSeenSkills(skillsBatch) = numel(seenSkills);
        numberOfLearnedSkills           = numel(seenSkills);
        
        for k = 1:parameters.totalSkillClusters
            SkillsInCluster(k,skillsBatch) = numel(intersect(seenSkills,parameters.skillClusters(:,k)));
        end
        parameters.clusterTransferrableKnowledgeFraction = SkillsInCluster(:,skillsBatch)./parameters.skillsPerCluster;
        disp(parameters.clusterTransferrableKnowledgeFraction)
        
        clusterTransferrableKnowledgeFraction_aux = [];
        for k = 1:parameters.totalSkillClusters
            clusterTransferrableKnowledgeFraction_aux = [clusterTransferrableKnowledgeFraction_aux, ...
                                                         repmat(parameters.clusterTransferrableKnowledgeFraction(k),1,parameters.numberOfRobotsPerCluster(k))];
        end
        parameters.clusterTransferrableKnowledgeFractionPerAgent = clusterTransferrableKnowledgeFraction_aux;
        % =========================================================================        


            

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
        if numberOfLearnedSkills == parameters.totalSkills
            warning('All skilles learned')
            break
        end
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

% =========================================================================
    
    figure('Color', 'w')
    plot(1:maxNumberOfProducts, numberOfNewSkills,'r')
    hold on
    plot(1:maxNumberOfProducts, numberOfSeenSkills,'b')
    plot(1:maxNumberOfProducts, SkillsInCluster,'k')
    % plot(skillsInCluster2,'k')
    % plot(skillsInCluster3,'k')
    % plot(skillsInCluster4,'k')
    legend('No. new skills','No. learned skills','Skills per cluster')
    xlabel('No. of products')
    ylabel('No. of skills learned')
    axis square


    results.c_jk_cl_dist_episodes = c_jk_cl_dist_episodes;
    results.learnedSkillsStorage  = learnedSkillsStorage;
    results.clusterKnowledge      = clusterKnowledge;
    results.numberOfNewSkills     = numberOfNewSkills;
    results.numberOfSeenSkills    = numberOfSeenSkills;
    results.SkillsInCluster      = SkillsInCluster;
end