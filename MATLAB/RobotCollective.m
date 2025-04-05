classdef RobotCollective < handle
    properties
        Alpha_max                   = 0.0691;
        Alpha_min                   = 0.0461;
        DISTRIBUTED_COLLECTIVE_LEARNING ...
                                    = boolean(1);          % Implies agents are distributed across clusters
        ClusterTransferableKnowledgeFraction ....
                                    = [];
        ClusterTransferableKnowledgeFractionPerAgent ....
                                    = [];
        ClusterSimilarityMatrix     = [];
        ClusterSimilarityMatrixPerAgent = [];
        Delta                       = 0.0360;        
        ENABLE_COLLECTIVE_LEARNING  = boolean(1);              % Enables collective learning: agent <-> agents knowledge sharing
        ENABLE_INCREMENTAL_LEARNING = boolean(1);              % Enables memory for an agent (incremental learning)
        ENABLE_TRANSFER_LEARNING    = boolean(1);              % Enables knowledge transfer (i.e., transfer learning)
        Episodes                    = 0:0.1:500;
        Eta_0                       = 0.1;
        Eta_std                     = 0.1;
        FundamentalComplexity       = 100;
        Gamma_0                     = 0.1;
        Gamma_std                   = 0.1;
        KnowledgeLowerBound         = 0.01;
        MaxNumberOfProducts         = [];
        MaxNumberOfRobots           = 128;
        MaxNumberOfSkillsPerProduct = 50;
        NumberOfRobots              = [];
        NumberOfRobotsPerCluster    = [];
        NumberOfSkillBatches        = [];
        NumberOfLearnedSkills       = 0;
        RAND_COMM_INTERRUPT         = boolean(1);              % Randomly cuts communication between pairs of agents
        REPEATING_SKILLS            = boolean(1);              % Controls if a given skills bacth conttais repeated skills
        SkillClusters               = [];
        SkillsInAgent               = [];
        SkillsInCluster             = [];
        SkillsPerCluster            = 128;
        TotalSimulationScenarios    = 5;
        TotalSkills                 = 512;
        TotalSkillClusters          = 4;
        UNEVEN_ROBOT_DISTRIBUTION   = boolean(1);
    end

    methods
        function obj = RobotCollective(NameValueArgs)
            arguments
                NameValueArgs.numberOfRobots           = [];
                NameValueArgs.numberOfRobotsPerCluster = []
                NameValueArgs.maxNumberOfProducts      = 500;
            end

            obj.NumberOfRobots = NameValueArgs.numberOfRobots;
            if isempty(NameValueArgs.numberOfRobotsPerCluster)
                obj.NumberOfRobotsPerCluster = repmat(obj.NumberOfRobots/obj.TotalSkillClusters,obj.TotalSkillClusters,1);
            end
            obj.ClusterTransferableKnowledgeFractionPerAgent = zeros(1,obj.NumberOfRobots);
            obj.MaxNumberOfProducts                          = NameValueArgs.maxNumberOfProducts;    
        
            if obj.REPEATING_SKILLS == 0
                if mod(obj.TotalSkills,obj.NumberOfRobots)==0
                    obj.NumberOfSkillBatches = obj.TotalSkills/obj.NumberOfRobots;
                    % Set a flag
                    obj.UNEVEN_ROBOT_DISTRIBUTION = 0;
                else
                    obj.NumberOfSkillBatches      = floor(obj.TotalSkills/obj.NumberOfRobots) + 1;
                    % Set a flag
                    obj.UNEVEN_ROBOT_DISTRIBUTION = 1;
                end
            else
                obj.NumberOfSkillBatches      = obj.MaxNumberOfProducts;
                % Set a flag
                obj.UNEVEN_ROBOT_DISTRIBUTION = 0; 
            end

            % Create skill clusters
            skillIndices  = randperm(obj.TotalSkills, obj.TotalSkills);
            skillClusters = reshape(skillIndices, obj.SkillsPerCluster, obj.TotalSkillClusters);
            for k = 1:obj.TotalSkillClusters
                sample_indices = randperm(obj.SkillsPerCluster,10);
                skillClusters(sample_indices,k);
            end
            obj.SkillClusters = skillClusters;
    
            % Define the similarity between clusters and agents
            rng("default")
            B = rand(obj.TotalSkillClusters);
            obj.ClusterSimilarityMatrix = 0.5*(triu(B,1) + transpose(triu(B,1)));

            ClusterSimilarityMatrixPerAgent     = [];% zeros(obj.NumberOfRobots,obj.NumberOfRobots);
            aux = eye(obj.TotalSkillClusters) + obj.ClusterSimilarityMatrix;
            for r=1:obj.NumberOfRobotsPerCluster
                ClusterSimilarityMatrixPerAgent_aux = [];
            
                theRow = aux(r,:);
                for k = 1:obj.TotalSkillClusters
                    ClusterSimilarityMatrixPerAgent_aux = [ClusterSimilarityMatrixPerAgent_aux, ...
                                                                 repmat(theRow(k),1,obj.NumberOfRobotsPerCluster(k))];
                end
                ClusterSimilarityMatrixPerAgent = [ClusterSimilarityMatrixPerAgent;repmat(ClusterSimilarityMatrixPerAgent_aux,obj.NumberOfRobotsPerCluster(r),1)];
            end
            obj.ClusterSimilarityMatrixPerAgent = ClusterSimilarityMatrixPerAgent;

            obj.ClusterTransferableKnowledgeFraction  = zeros(obj.TotalSkillClusters,1);

        end
        
        % The dynamics of Isolated Learning (IsL)
        function out = isolatedLearningRemainingKnowledgeDynamics(obj)
            arguments
                obj (1,1) RobotCollective
            end            
            Alpha = diag(obj.Alpha_min + (obj.Alpha_max - obj.Alpha_min).*rand(obj.NumberOfRobots,1));
            out   = Alpha;
        end
        
        % The dynamics of Incremental Learning (IL)
        function out = incrementalLearningRemainingKnowledgeDynamics(obj, NameValueArgs)
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.eta
            end    
            if obj.ENABLE_INCREMENTAL_LEARNING == 1
                out = (NameValueArgs.eta.*obj.NumberOfLearnedSkills + 1);
            else
                out = 1;
            end
        end

        % The initial condition for Incremental Learning
        function out = incrementalLearningInitialRemainingKnowledge(obj) 
            arguments
                obj (1,1) RobotCollective
            end
            out = exp(-obj.Delta*obj.NumberOfLearnedSkills.*double(obj.ENABLE_INCREMENTAL_LEARNING));   
        end

        % Learning loop for a batch of skills
        function [c_jk_cl_dist_episodes, learnedSkillsStorage, clusterKnowledge, results] = ...
            simulateDistributedCollectiveKnowledgeDynamics(obj, NameValueArgs)
        
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.eta_0               = [];
                NameValueArgs.gamma_0             = [];
                NameValueArgs.maxNumberOfProducts = [];
            end
        
            if ~isempty(NameValueArgs.eta_0)
                % Use default value
                obj.Eta_0 = NameValueArgs.eta_0;
            end
            
            if ~isempty(NameValueArgs.gamma_0)
                % Use default value
                obj.Gamma_0 = NameValueArgs.gamma_0;
            end

            if ~isempty(NameValueArgs.gamma_0)
                % Use default value
                obj.MaxNumberOfProducts = NameValueArgs.maxNumberOfProducts;
            end      
        
            % Create figure to track progress =============================
            fig = figure('color','w');
            p   = area(obj.Episodes, obj.KnowledgeLowerBound*ones(numel(obj.Episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            hold on
            plot(obj.Episodes,obj.KnowledgeLowerBound*ones(numel(obj.Episodes),1),'k:','LineWidth',2);
            % set(gca, 'YScale', 'log')
            % set(gca, 'XScale', 'log')
            % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel('$\bar{\boldmath{\sigma}}^{(\mathrm{DCL})}_{j,k}$','FontSize',25,'Interpreter','latex')
            % Generate a colormap (e.g., 'parula', 'jet', 'hot', 'cool', etc.)
            cmap = colormap('lines');
            % Create indices to pick colors evenly from the colormap
            colorIndices = linspace(1, size(cmap, 1), obj.NumberOfSkillBatches);
            % Interpolate the colormap to get the desired number of colors
            selectedColors = interp1(1:size(cmap, 1), cmap, colorIndices);
            % Format y-axis
            ylim([0.008 1E0])
            yticks([1E-3 1E-2 1E-1 1E0])
            yticklabels({'','$\epsilon$', '', '1'})
            set(gca,'TickLabelInterpreter','latex')
            hold on            
            % =============================================================
            
        
            % NOTE: Subindex jk means SKILL j in CLUSTER k
            complexity_jk_CL_Distributed = zeros(obj.NumberOfSkillBatches, 1);
            clusterKnowledge             = zeros(obj.NumberOfSkillBatches, 1);
            
            % Basics learning rates PER agent (embodiment dependent)
            % Alpha = mean([obj.Alpha_min, obj.Alpha_max])*eye(obj.NumberOfRobots);
            Alpha = obj.isolatedLearningRemainingKnowledgeDynamics(); % diag(obj.Alpha_min + (obj.Alpha_max - obj.Alpha_min).*rand(obj.NumberOfRobots,1));
            
            % Initialization 
            seenSkills          = []; % Pool of learned skills 
            numberOfNewSkills   = zeros(1,obj.MaxNumberOfProducts);
            numberOfSeenSkills  = zeros(1,obj.MaxNumberOfProducts);
            obj.SkillsInCluster = zeros(obj.TotalSkillClusters,obj.MaxNumberOfProducts);
            
            % Loop over skill batches (products) ==========================
            for skillsBatch = 1:obj.NumberOfSkillBatches
        
                % NOTE: the transferrableKnowledgeFraction (i.e., beta_k) varies at
                %       every cycle. From each cluster, only a fraction of knowlede
                %       can be transfered.
                % beta_k = obj.numberOfRobots*learnedSkills/obj.totalSkills;
                

                scaledTransferableKnowledge = min(0.99,sum(obj.ClusterSimilarityMatrix*(obj.ClusterTransferableKnowledgeFraction),2)).*double(obj.ENABLE_TRANSFER_LEARNING);
                clusterTransferableKnowledgeFraction_aux = [];
                for k = 1:obj.TotalSkillClusters
                    clusterTransferableKnowledgeFraction_aux = [clusterTransferableKnowledgeFraction_aux, ...
                                                                 repmat(scaledTransferableKnowledge(k),1,obj.NumberOfRobotsPerCluster(k))];
                end
                obj.ClusterTransferableKnowledgeFractionPerAgent = clusterTransferableKnowledgeFraction_aux;
% skillTransferrableKnowledgeFraction2 = (obj.NumberOfLearnedSkills/obj.TotalSkills)*double(obj.ENABLE_TRANSFER_LEARNING);        


% disp(obj.ClusterTransferableKnowledgeFraction)
% 
% clusterTransferableKnowledgeFraction_aux = [];
% for k = 1:obj.TotalSkillClusters
%     clusterTransferableKnowledgeFraction_aux = [clusterTransferableKnowledgeFraction_aux, ...
%                                                  repmat(obj.ClusterTransferableKnowledgeFraction(k),1,obj.NumberOfRobotsPerCluster(k))];
% end
% obj.ClusterTransferableKnowledgeFractionPerAgent = clusterTransferableKnowledgeFraction_aux;








                skillTransferableKnowledgeFraction = mean((obj.NumberOfLearnedSkills./obj.TotalSkills)*double(obj.ENABLE_TRANSFER_LEARNING));        
                
                
                clusterKnowledge(skillsBatch)       = mean(skillTransferableKnowledgeFraction);
                j      = skillsBatch;%learnedSkills + 1;
                fprintf('Gamma scenario: %2i | Eta scenario: %2i | Skills batch: %2i/%3i | Cluster knowledge: %1.3f \n',1, 1, skillsBatch, obj.NumberOfSkillBatches, skillTransferableKnowledgeFraction)
                
                if (obj.UNEVEN_ROBOT_DISTRIBUTION == 0)
                    % initialRemainingKnowledge = (1 - skillTransferableKnowledgeFraction)*obj.incrementalLearningInitialRemainingKnowledge().*ones(obj.NumberOfRobots,1);
                    initialRemainingKnowledge = diag(1 - obj.ClusterTransferableKnowledgeFractionPerAgent)*(obj.incrementalLearningInitialRemainingKnowledge().*ones(obj.NumberOfRobots,1));




                    % fprintf('Gamma scenario: %2i | Eta scenario: %2i | Skills batch: %2i/%3i | IR knowledge: %1.3f \n',1, 1, skillsBatch, obj.NumberOfSkillBatches, mean(initialRemainingKnowledge))
                    % remainingKnowledge = transpose( ...
                    %     ode4_sat(@(n,sigmaBar) ...
                    %         obj.collectiveKnowledgeSharingDynamicsEpisodic( ...
                    %             agentsRemainingKnowledge = sigmaBar, ...
                    %             isolatedLearningRemainingKnowledgeDynamics = Alpha, ...
                    %             totalTrasferableKnowledgeFraction = skillTransferableKnowledgeFraction), obj.Episodes, initialRemainingKnowledge)); 
                    remainingKnowledge = transpose( ...
                        ode4_sat(@(n,sigmaBar) ...
                            obj.collectiveKnowledgeSharingDynamicsEpisodic( ...
                                agentsRemainingKnowledge = sigmaBar, ...
                                isolatedLearningRemainingKnowledgeDynamics = Alpha), obj.Episodes, initialRemainingKnowledge)); 

                elseif (obj.UNEVEN_ROBOT_DISTRIBUTION == 1) 
                    warning('Code under revision')
                    % if (skillsBatch < obj.NumberOfSkillBatches)
                    %     initialRemainingKnowledge = (1 - totalTransferrableKnowledgeFraction)*obj.initialCondition().*ones(obj.NumberOfRobots,1);
                    %     remainingKnowledge = transpose( ...
                    %         ode4_sat(@(n,sigmaBar) ...
                    %             fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                    %                 n, sigmaBar, ...
                    %                 totalTransferrableKnowledgeFraction), obj.Episodes, initialRemainingKnowledge));       
        
                    % elseif (skillsBatch == obj.NumberOfSkillBatches)
                    %     initialRemainingKnowledge = (1 - totalTransferrableKnowledgeFraction)*obj.initialCondition().*ones(obj.TotalSkills-obj.NumberOfRobots*(obj.NumberOfSkillBatches-1),1);
                    %     remainingKnowledge = transpose( ...
                    %         ode4_sat(@(n,sigmaBar) ...
                    %             fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                    %                 n, sigmaBar, ...
                    %                 obj.totalSkills-obj.numberOfRobots*(numberOfSkillBatches-1), ...
                    %                 Alpha(1:obj.totalSkills-obj.numberOfRobots*(numberOfSkillBatches-1),1:obj.totalSkills-obj.numberOfRobots*(numberOfSkillBatches-1)), ...
                    %                 totalTransferrableKnowledgeFraction, ...
                    %                 obj, n), obj.episodes, initialRemainingKnowledge));  
                    % end
                end
                complexity_jk_CL_Distributed(j) = ...
                    ceil( ...
                        mean( ...
                            obj.Episodes( ...
                                cell2mat( ...
                                    arrayfun( ...
                                        @(i) ( ...
                                            find( ...
                                                remainingKnowledge(i,:)<obj.KnowledgeLowerBound,1,'first')),1:size(remainingKnowledge,1),'UniformOutput',false)))));

                if isnan(complexity_jk_CL_Distributed(j))
                    complexity_jk_CL_Distributed(j) = obj.FundamentalComplexity;
                end                
                
                % Trigger warning if there are unlearned skills
                if any(remainingKnowledge(:,end)>obj.KnowledgeLowerBound)
                    warning('Some skills COULD NOT be learned')
                end    
        
                % mean(initialRemainingKnowledge)
                % Add the learned skills in the batch to the general pool of
                % learned skills
                % numberOfLearnedSkills = numberOfLearnedSkills + sum(remainingKnowledge(:,end)<obj.knowledgeLowerBound);
        
        
                % =========================================================================
                productSkills = randi(obj.TotalSkills,1,obj.MaxNumberOfSkillsPerProduct);
                if obj.MaxNumberOfSkillsPerProduct<obj.NumberOfRobots
                    extraSkills   = productSkills(randi(numel(productSkills),1,obj.NumberOfRobots-obj.MaxNumberOfSkillsPerProduct));
                    productSkills = [productSkills, extraSkills];
                end
          


                obj.SkillsInAgent = [obj.SkillsInAgent,productSkills'];%numel(unique([seenSkills, productSkills])) - numel(seenSkills);


                numberOfNewSkills(skillsBatch)  = numel(unique([seenSkills, productSkills])) - numel(seenSkills);
                seenSkills                      = unique([seenSkills, productSkills]);
                numberOfSeenSkills(skillsBatch) = numel(seenSkills);
                if obj.ENABLE_COLLECTIVE_LEARNING == 1
                    obj.NumberOfLearnedSkills  = numel(seenSkills).*ones(obj.NumberOfRobots,1);
                else
                    obj.NumberOfLearnedSkills = reshape(arrayfun(@(i) numel(unique(obj.SkillsInAgent(i,:))),1:obj.NumberOfRobots),obj.NumberOfRobots,1);
                end

                for k = 1:obj.TotalSkillClusters
                    obj.SkillsInCluster(k,skillsBatch) = numel(intersect(seenSkills,obj.SkillClusters(:,k)));
                end
                obj.ClusterTransferableKnowledgeFraction = obj.SkillsInCluster(:,skillsBatch)./obj.SkillsPerCluster;
                disp(obj.ClusterTransferableKnowledgeFraction)
                
                clusterTransferableKnowledgeFraction_aux = [];
                for k = 1:obj.TotalSkillClusters
                    clusterTransferableKnowledgeFraction_aux = [clusterTransferableKnowledgeFraction_aux, ...
                                                                 repmat(obj.ClusterTransferableKnowledgeFraction(k),1,obj.NumberOfRobotsPerCluster(k))];
                end
                obj.ClusterTransferableKnowledgeFractionPerAgent = clusterTransferableKnowledgeFraction_aux;
                % =========================================================================        
        
                % Determine the complexity
                % complexity_jk_CL_Distributed(j) = ...
                %     ceil( ...
                %         mean( ...
                %             obj.Episodes( ...
                %                 cell2mat( ...
                %                     arrayfun( ...
                %                         @(i) min( ...
                %                             find( ...
                %                                 remainingKnowledge(i,:)<obj.KnowledgeLowerBound)),1:size(remainingKnowledge,1),'UniformOutput',false)))));


                % complexity_jk_CL_Distributed(j) = ...
                %     ceil( ...
                %         mean( ...
                %             obj.Episodes( ...
                %                 cell2mat( ...
                %                     arrayfun( ...
                %                         @(i) ( ...
                %                             find( ...
                %                                 remainingKnowledge(i,:)<obj.KnowledgeLowerBound,1,'first')),1:size(remainingKnowledge,1),'UniformOutput',false)))));

                % if isnan(complexity_jk_CL_Distributed(j))
                %     complexity_jk_CL_Distributed(j) = obj.FundamentalComplexity;
                % end
        
                % loglog(obj.episodes,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
                % loglog(obj.episodes,(remainingKnowledge),'LineStyle','-','color',[0.7 0.7 0.7],'LineWidth',1)
                % xlim([obj.episodes(1), obj.episodes(end)])
                % xticks([10^0 10^1 10^2])
                % xticklabels({'1','10', '$c_0$'})
        
        
                % gammaValues = 0.022*obj.episodes - 0.2;
                % % semilogy(gammaValues,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
                % % semilogy(gammaValues,remainingKnowledge,'LineStyle','-','color',rand(1,3),'LineWidth',1)
                % loglog(obj.episodes,remainingKnowledge,'LineStyle','-','color',selectedColors(skillsBatch,:),'LineWidth',1)
                
                
                % Plot remaining knowledge curves for the skill batch ==========================
                % Calculate mean and standard deviation across all curves
                meanCurve = mean(remainingKnowledge,1);
                stdCurve  = std(remainingKnowledge, 0, 1);
                
                % Define the upper and lower bounds of the shaded region
                upperBound = meanCurve + stdCurve;
                lowerBound = max(obj.KnowledgeLowerBound,meanCurve - stdCurve);
                
                % Create the plot
                % Plot the shaded region for standard deviation
                if ~all(meanCurve < obj.KnowledgeLowerBound)
                    patch([obj.Episodes, fliplr(obj.Episodes)], [upperBound, fliplr(lowerBound)], selectedColors(skillsBatch,:), ...
                        'EdgeColor', 'none', 'FaceAlpha', 0.3);
                end
                % Plot the mean curve
                plot(obj.Episodes, meanCurve, 'b-', 'LineWidth', 0.5,'Color',selectedColors(skillsBatch,:));
                set(gca, 'YScale', 'log')
                set(gca, 'XScale', 'log')
                plot(obj.FundamentalComplexity*ones(100,1), linspace(obj.KnowledgeLowerBound,1,100),'k--')
                % =========================================================================        
                
                % xlim([obj.Episodes(1) obj.Episodes(end)])
                % % xlim([gammaValues(1) gammaValues(end)])
                % xticks(obj.episodes([1,11,101,1001,5001]))
                % xticklabels(arrayfun(@(i) num2str(gammaValues(i)),[1,11,101,1001,5001],'UniformOutput',false))        
                
                % % Format y-axis
                % ylim([0.008 1E0])
                % yticks([1E-3 1E-2 1E-1 1E0])
                % yticklabels({'','$\epsilon$', '', '1'})
                % set(gca,'TickLabelInterpreter','latex')
                % hold on
                % exportgraphics(gcf,'always_stable.gif','Append',true,'Resolution',300)
                pause(1)
                if obj.NumberOfLearnedSkills == obj.TotalSkills
                    warning('All skilles learned')
                    break
                end
            end
              
        
            % p = area(obj.episodes,  obj.knowledgeLowerBound*ones(numel(obj.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            % plot(obj.episodes,obj.knowledgeLowerBound*ones(numel(obj.episodes),1),'k:','LineWidth',2);
            % % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
            % xlabel('Episodes','FontSize',25)
            % ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CLd})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
            title(gca,['$m =',num2str(obj.NumberOfRobots),'~|~\bar{\eta}=',num2str(obj.Eta_0),'~|~\bar{\gamma} =', num2str(obj.Gamma_0),'$'],'Interpreter','latex','FontSize',11)
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
        %         xlim(ax2, [0 obj.episodes(end)])
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
            learnedSkillsStorage  = obj.NumberOfLearnedSkills;
        
        % =========================================================================
            
            figure('Color', 'w')
            plot(1:obj.MaxNumberOfProducts, numberOfNewSkills,'r')
            hold on
            plot(1:obj.MaxNumberOfProducts, numberOfSeenSkills,'b')
            plot(1:obj.MaxNumberOfProducts, obj.SkillsInCluster,'k')
            legend('No. new skills','No. learned skills','Skills per cluster')
            xlabel('No. of products')
            ylabel('No. of skills learned')
            axis square
        
            % Put together <results> structure
            results.c_jk_cl_dist_episodes = c_jk_cl_dist_episodes;
            results.learnedSkillsStorage  = learnedSkillsStorage;
            results.clusterKnowledge      = clusterKnowledge;
            results.numberOfNewSkills     = numberOfNewSkills;
            results.numberOfSeenSkills    = numberOfSeenSkills;
            results.obj.SkillsInCluster      = obj.SkillsInCluster;
        end
    
        
        % Remaining knowledge dynamics functiion
        function d_coupledRemainingKnowledge = collectiveKnowledgeSharingDynamicsEpisodic( ...
                obj, ...
                NameValueArgs)
           
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.isolatedLearningRemainingKnowledgeDynamics % singleAgentLearningRates
                NameValueArgs.agentsRemainingKnowledge
                NameValueArgs.totalTrasferableKnowledgeFraction
            end
            isolatedLearningRemainingKnowledgeDynamics = NameValueArgs.isolatedLearningRemainingKnowledgeDynamics;
            agentsRemainingKnowledge                   = NameValueArgs.agentsRemainingKnowledge;
            % totalTrasferableKnowledgeFraction          = NameValueArgs.totalTrasferableKnowledgeFraction;

            eta_robots             = obj.Eta_0 + obj.Eta_std.*randn(obj.NumberOfRobots,1);    
            % selfKnowledgeDynamics  = -isolatedLearningRemainingKnowledgeDynamics.*((1 - totalTrasferableKnowledgeFraction).^(-1)).*obj.incrementalLearningRemainingKnowledgeDynamics(eta= eta_robots).*eye(obj.NumberOfRobots);
            selfKnowledgeDynamics  = -isolatedLearningRemainingKnowledgeDynamics.*diag((1 - obj.ClusterTransferableKnowledgeFractionPerAgent).^(-1)).*obj.incrementalLearningRemainingKnowledgeDynamics(eta = eta_robots).*eye(obj.NumberOfRobots);
        
        % numberOfRobotsPerCluster = numberOfRobots/obj.totalSkillClusters;
        % b = (clusterTrasferrableKnowledgeFraction-1)*((numberOfRobotsPerCluster-1) + (obj.totalSkillClusters-1)*numberOfRobotsPerCluster*clusterTrasferrableKnowledgeFraction);
        % a = Alpha(1,1);
        % fun = mean(f(eta_robots, numberOfLearnedSkills));
        % % obj.gamma_0 =  (a/b)*fun + 2*abs((a/b)*fun);
        % disp([obj.gamma_0, a, b, (a/b), (a/b)*fun, numberOfLearnedSkills, obj.gamma_0 > (a/b)*fun])
        % 
        % if ~(obj.gamma_0 > (a/b)*fun)
        %     warning('Condition violated')
        % end
        
        
            % numberOfRobotsPerCluster     = numberOfRobots/obj.totalSkillClusters;
            integratedKnowledgeDynamics  = zeros(obj.NumberOfRobots,1);
            weightedAdjacencyMatrix      = obj.generateGamma();
        
            % If collective learning is distributed, the <weightedAdjacencyMatrix>
            % needs to be scaled down by the cluster knowledge fraction transfer
            % factor beta_k
        %     if obj.DISTRIBUTED_COLLECTIVE_LEARNING == 1
        %         adjacencyMatrix       = double(weightedAdjacencyMatrix~=0);
        % 
        % 
        % % totalTrasferableKnowledgeFraction
        % clusterTransferMatrix = totalTrasferableKnowledgeFraction*adjacencyMatrix;
        % % clusterTransferMatrix = repmat(obj.ClusterTransferableKnowledgeFractionPerAgent,obj.NumberOfRobots,1).*adjacencyMatrix;
        % 
        %         for i = 1:obj.TotalSkillClusters
        %            clusterTransferMatrix((obj.NumberOfRobots/obj.TotalSkillClusters)*(i-1) + 1:(obj.NumberOfRobots/obj.TotalSkillClusters)*i,(obj.NumberOfRobots/obj.TotalSkillClusters)*(i-1) + 1:(obj.NumberOfRobots/obj.TotalSkillClusters)*i) = ones(obj.NumberOfRobots/obj.TotalSkillClusters) - eye(obj.NumberOfRobots/obj.TotalSkillClusters);
        %         end
        % 
        % clusterTransferMatrix                    = triu(clusterTransferMatrix) + transpose(triu(clusterTransferMatrix));
        % 
        % clusterTransferMatrix = obj.ClusterSimilarityMatrixPerAgent - eye(obj.NumberOfRobots);
        % 
        % perturbation_B                           = rand(obj.NumberOfRobots,obj.NumberOfRobots);
        %         perturbation_B(clusterTransferMatrix==0) = 1;
        %         perturbation_B(clusterTransferMatrix==1) = 1;
        %         perturbation_B                           = (perturbation_B + transpose(perturbation_B))./2; % Ensure the matrix is symmetric
        %         clusterTransferMatrix                    = clusterTransferMatrix.*perturbation_B;
        % 
        % % if sum(clusterTransferMatrix-clusterTransferMatrix','all') ~= 0
        % %     warning('Error in B matrix')
        % % end
        % 
        %         weightedAdjacencyMatrix = clusterTransferMatrix.*weightedAdjacencyMatrix;
        %     end
            
            if obj.ENABLE_COLLECTIVE_LEARNING == 0 && obj.DISTRIBUTED_COLLECTIVE_LEARNING == 1
                clusterTransferMatrix                    = obj.ClusterSimilarityMatrixPerAgent - eye(obj.NumberOfRobots);
                        
                perturbation_B                           = rand(obj.NumberOfRobots,obj.NumberOfRobots);
                perturbation_B(clusterTransferMatrix==0) = 1;
                perturbation_B(clusterTransferMatrix==1) = 1;
                perturbation_B                           = (perturbation_B + transpose(perturbation_B))./2; % Ensure the matrix is symmetric
                
                clusterTransferMatrix                    = clusterTransferMatrix.*perturbation_B;
                        
                weightedAdjacencyMatrix = clusterTransferMatrix.*weightedAdjacencyMatrix;
            end

            % In agents are not sharing knowledge, cancel the connectivity matrix
            if obj.ENABLE_COLLECTIVE_LEARNING == 0
                weightedAdjacencyMatrix = 0*weightedAdjacencyMatrix;
            end
        
            % =====================================================================
        
            for agent = 1:obj.NumberOfRobots
                knowledgeIntegrationVector         = obj.knowledgeIntegration(agentsRemainingKnowledge,agentsRemainingKnowledge(agent));
                integratedKnowledgeDynamics(agent) = ...
                    weightedAdjacencyMatrix(agent,:)*knowledgeIntegrationVector;
            end
        
            mainDynamicsCoeff  = (diag(selfKnowledgeDynamics) - integratedKnowledgeDynamics);
            % if any(mainDynamicsCoeff > 0)
            %     disp('Knowledge corruption!')
            % end
            d_coupledRemainingKnowledge = mainDynamicsCoeff.*agentsRemainingKnowledge;
            % d_coupledRemainingKnowledge = -TheMatrix*agentsRemainingKnowledge
            
            % if any(abs(d_coupledRemainingKnowledge_1 - d_coupledRemainingKnowledge)>1E-6)
            %     bbb= 0;
            % end
            
            % Constraints to stop the knowledge evolution when the minimum
            % threshold is reached
            d_coupledRemainingKnowledge(agentsRemainingKnowledge < obj.KnowledgeLowerBound) = 0;
            % d_coupledRemainingKnowledge(agentsRemainingKnowledge>1) = 0;
        end
        
        function distance = knowledgeIntegration(obj, neighborsKnowledge,agentKnowledge)
            alpha    = 2;
            % distance = exp(-alpha*abs(neighborsKnowledge-agentKnowledge));
            distance = exp(-alpha*(neighborsKnowledge-agentKnowledge).^2);
        end
               
        function Gamma = generateGamma(obj)
            if obj.RAND_COMM_INTERRUPT == 1
                commInterruptMatrix = randn(obj.NumberOfRobots,obj.NumberOfRobots)>-0.5;
                commInterruptMatrix = (commInterruptMatrix + transpose(commInterruptMatrix))./2;
            else
                commInterruptMatrix = ones(obj.NumberOfRobots,obj.NumberOfRobots);
            end
            
            % Gamma   = pearsrnd(gamma_mean, gamma_std, -1.5, 10, numberOfRobots,numberOfRobots);
            Gamma   = obj.Gamma_0 + obj.Gamma_std.*randn(obj.NumberOfRobots,obj.NumberOfRobots);
            Gamma   = (Gamma + transpose(Gamma))./2;
            Gamma   = Gamma - diag(Gamma).*eye(obj.NumberOfRobots);
            Gamma   = Gamma.*commInterruptMatrix;
            % Gamma   = Gamma.*(full(sprandsym(Gamma))>-0.5);
        end        
    end

    

end