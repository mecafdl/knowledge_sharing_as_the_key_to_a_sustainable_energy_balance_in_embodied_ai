% %% - Functions folder in the same folder
% clc
% clearvars
% 
% cd(fileparts(matlab.desktop.editor.getActiveFilename)); % use this to CD
% addpath(genpath(pwd))
% 
% % Threshold to consider a skill learned
% parameters.knowledgeLowerBound = 0.01;
% 
% % Fundamental complexity (AKA max number of parameters.episodes to learn any skill)
% parameters.fundamentalComplexity = 100;
% % Trial parameters.episodes
% % parameters.episodes  = 0:0.01:parameters.fundamentalComplexity; 
% parameters.episodes  = 0:0.1:parameters.fundamentalComplexity; 
% 
% % Total number of skills
% parameters.totalSkills = 8*64;
% 
% % Number of clusters
% parameters.totalSkillClusters = 4;
% 
% % Number of skills per cluster
% parameters.skillsPerCluster = parameters.totalSkills/parameters.totalSkillClusters;
% 
% % Base learning rate
% parameters.alpha_min = -log(parameters.knowledgeLowerBound)/parameters.fundamentalComplexity;%0.05;
% parameters.alpha_max = 1.5*parameters.alpha_min;
% 
% parameters.delta = -log(parameters.knowledgeLowerBound)/parameters.skillsPerCluster;%0.05;
% 
% parameters.eta_0   = 0.1;
% parameters.eta_std = 0.3*parameters.eta_0;
% 
% parameters.gamma_0   = 4*0.05;
% parameters.gamma_std = 0.5*parameters.gamma_0;
% 
% f = @(eta, N_zeta) eta.*N_zeta+1;
% g = @(delta, N_zeta) exp(-delta*N_zeta);
% 
% % number of robots available
% parameters.maxNumberOfRobots = 128;
% parameters.totalSimulationScenarios = 5;


classdef RobotCollective < handle
    properties
        Alpha_max                   = 0.0691;
        Alpha_min                   = 0.0461;
        AgentMemory                 = [];
        AgentStorage                 = [];
        DISTRIBUTED_COLLECTIVE_LEARNING ...
                                    = boolean(1);              % Implies agents are unevenly distributed across clusters
        ClusterTransferableKnowledgeFraction ....
                                    = [];
        AggregatedScaledClusterTransferableKnowledgeFractionPerAgent ....
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
        MaxNumberOfSkillsPerProduct = 8;
        NumberOfNewSkills           = [];
        NumberOfRobots              = [];
        NumberOfRobotsPerCluster    = [];
        NumberOfSeenSkills          = [];
        NumberOfSkillBatches        = [];
        NumberOfLearnedSkillsPerAgentAndTargetSkill ...
                                    = [];                      % Refers to the in-memory learned skills that can be used to learn the skill currently executed by each agent
        ClusterTransferableKnowledgeFractionPerAgentAndTargetSkill ...
                                    = [];                      % Contains the cluster knowledge fraction in memory for each agent
        RAND_COMM_INTERRUPT         = boolean(1);              % Randomly cuts communication between pairs of agents
        REPEATING_SKILLS            = boolean(1);              % Controls if a given skills bacth conttais repeated skills
        SeenSkills                  = [];
        SkillClusters               = [];
        SkillClusterMembership      = [];
        SkillsInAgent               = [];
        SkillsInCluster             = [];
        SkillsPerCluster            = 128;
        SkillsRemainingKnowledge    = [];
        TotalSimulationScenarios    = 5;
        TotalSkills                 = 512;
        TotalSkillClusters          = 4;
        UNEVEN_ROBOT_DISTRIBUTION   = boolean(1);
    end

    methods
        %% Initialization function
        function obj = RobotCollective(NameValueArgs)
            arguments
                NameValueArgs.numberOfRobots           = [];
                NameValueArgs.numberOfRobotsPerCluster = []
                NameValueArgs.maxNumberOfProducts      = 1000;
                NameValueArgs.repeatingSkills          = false;
                NameValueArgs.MaxNumberOfSkillsPerProduct
            end
            obj.MaxNumberOfSkillsPerProduct = NameValueArgs.MaxNumberOfSkillsPerProduct;
        
            obj.NumberOfRobots = NameValueArgs.numberOfRobots;
            if isempty(NameValueArgs.numberOfRobotsPerCluster)
                obj.NumberOfRobotsPerCluster = repmat(obj.NumberOfRobots/obj.TotalSkillClusters,obj.TotalSkillClusters,1);
            end
            obj.AggregatedScaledClusterTransferableKnowledgeFractionPerAgent = zeros(1,obj.NumberOfRobots);
            obj.MaxNumberOfProducts                          = NameValueArgs.maxNumberOfProducts;    
        
            % Initialize the number of learned skills by an agent pertaining
            % the skill currently being learned
            obj.NumberOfLearnedSkillsPerAgentAndTargetSkill= zeros(obj.NumberOfRobots,1);

            if NameValueArgs.repeatingSkills == 0
                obj.REPEATING_SKILLS = NameValueArgs.repeatingSkills;

                % obj.NumberOfSkillBatches = obj.TotalSkills/obj.NumberOfRobots;
                % if mod(obj.TotalSkills,obj.NumberOfRobots)==0
                %     obj.NumberOfSkillBatches = obj.TotalSkills/obj.NumberOfRobots;
                %     % Set a flag
                %     obj.UNEVEN_ROBOT_DISTRIBUTION = 0;
                % else
                %     obj.NumberOfSkillBatches      = floor(obj.TotalSkills/obj.NumberOfRobots) + 1;
                %     % Set a flag
                %     obj.UNEVEN_ROBOT_DISTRIBUTION = 1;
                % end
                if mod(obj.TotalSkills,obj.MaxNumberOfSkillsPerProduct)==0
                    obj.NumberOfSkillBatches = obj.TotalSkills/obj.MaxNumberOfSkillsPerProduct;
                    % Set a flag
                    obj.UNEVEN_ROBOT_DISTRIBUTION = 0;
                else
                    obj.NumberOfSkillBatches      = floor(obj.TotalSkills/obj.MaxNumberOfSkillsPerProduct) + 1;
                    % Set a flag
                    obj.UNEVEN_ROBOT_DISTRIBUTION = 1;
                end                
            else
                obj.NumberOfSkillBatches      = obj.MaxNumberOfProducts;
                % Set a flag
                obj.UNEVEN_ROBOT_DISTRIBUTION = 0; 
            end

            obj.SkillsRemainingKnowledge = ones(obj.TotalSkills,1);

            % Create skill clusters
            skillIndices  = randperm(obj.TotalSkills, obj.TotalSkills);
            skillClusters = reshape(skillIndices, obj.SkillsPerCluster, obj.TotalSkillClusters);
            for k = 1:obj.TotalSkillClusters
                sample_indices = randperm(obj.SkillsPerCluster,10);
                skillClusters(sample_indices,k);
            end
            obj.SkillClusters = skillClusters;
    
            % Define the similarity between clusters and agents
            clusteSimilarityRandomNumberStream = RandStream.create('shr3cong');
            reset(clusteSimilarityRandomNumberStream,1)
            B = rand(clusteSimilarityRandomNumberStream, obj.TotalSkillClusters);
            obj.ClusterSimilarityMatrix = 0.5*(triu(B,1) + transpose(triu(B,1)));

            % Initialize the trasferable knowledge fraction
            obj.ClusterTransferableKnowledgeFraction = zeros(obj.TotalSkillClusters,1);
            obj.ClusterTransferableKnowledgeFractionPerAgentAndTargetSkill = zeros(obj.TotalSkillClusters,obj.NumberOfRobots);
            
            % Initialize the memory
            obj.AgentMemory = cell(1,obj.NumberOfRobots);

            % Initialize the each agent's storage
            obj.AgentStorage = cell(obj.NumberOfRobots,obj.TotalSkillClusters);
            obj.AgentStorage(:) = {NaN};
        end
        
        %% Generate plot canvas
        function [fig, ax, selectedColors] = generateCanvas(obj)
            fig = figure('color','w');
            p   = area(obj.Episodes, obj.KnowledgeLowerBound*ones(numel(obj.Episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            hold on
            plot(obj.Episodes,obj.KnowledgeLowerBound*ones(numel(obj.Episodes),1),'k:','LineWidth',2);
            set(gca, 'YScale', 'log')
            set(gca, 'XScale', 'log')

            xlabel('Episodes','FontSize',25)

            learningParadigm = [obj.ENABLE_INCREMENTAL_LEARNING, ...
                                obj.ENABLE_TRANSFER_LEARNING, ...
                                obj.ENABLE_COLLECTIVE_LEARNING];
            learningCasesLabels = {'IsL','IL','TIL','DCL'};
            if all(learningParadigm == [0 0 0])
                learningParadigmLabel = learningCasesLabels{1};
            elseif all(learningParadigm == [1 0 0])
                learningParadigmLabel = learningCasesLabels{2};    
            elseif all(learningParadigm == [1 1 0])
                learningParadigmLabel = learningCasesLabels{3};    
            elseif all(learningParadigm == [1 1 1])
                learningParadigmLabel = learningCasesLabels{4};
            end

            ylabel(sprintf("$\\bar{\\boldmath{\\sigma}}^{(\\mathrm{%s})}_{j,k}$",learningParadigmLabel),'FontSize',25,'Interpreter','latex')
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
            title(gca,['$m =',num2str(obj.NumberOfRobots),'~|~\bar{\eta}=',num2str(obj.Eta_0),'~|~\bar{\gamma} =', num2str(obj.Gamma_0),'$'],'Interpreter','latex','FontSize',11)


% theString = sprintf("$N_r = %i ~|~ $ \\bar{\\eta}=%f \\pm %f ~|~\\bar{\\gamma} = %f \\pm %f$",obj.NumberOfRobots,obj.Eta_0,obj.Eta_std,obj.Gamma_0, obj.Gamma_std);
% title(gca,theString,'Interpreter','latex','FontSize',11)
title(gca,['$N_r =',num2str(obj.NumberOfRobots),'~|~\bar{\eta}=',num2str(obj.Eta_0),'~|~\bar{\gamma} =', num2str(obj.Gamma_0),'$'],'Interpreter','latex','FontSize',11)

            hold(gca,"on")
            ax = gca;
        end

        %% Knowledge plots
        function plotRemainingKnowledgeCurves(obj, NameValueArgs)
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.remainingKnowledge
                NameValueArgs.skillsBatch
                NameValueArgs.figureHandle
                NameValueArgs.axisHandle
                NameValueArgs.lineColors
            end   
            remainingKnowledge = NameValueArgs.remainingKnowledge;
            skillsBatch        = NameValueArgs.skillsBatch;
            fig                = NameValueArgs.figureHandle;
            ax                 = NameValueArgs.axisHandle;
            selectedColors     = NameValueArgs.lineColors;
            
            % Plot remaining knowledge curves for the skill batch
            remainingKnowledge(remainingKnowledge(:,1)<obj.KnowledgeLowerBound,:) = [];

            meanCurve = mean(remainingKnowledge,1);
            stdCurve  = std(remainingKnowledge, 0, 1);
            
            % Define the upper and lower bounds of the shaded region
            upperBound = meanCurve + stdCurve;
            lowerBound = max(obj.KnowledgeLowerBound,meanCurve - stdCurve);
            
            % Create the plot
            % Plot the shaded region for standard deviation
            if ~all(meanCurve < obj.KnowledgeLowerBound)
                patch(ax, [obj.Episodes, fliplr(obj.Episodes)], [upperBound, fliplr(lowerBound)], selectedColors(skillsBatch,:), ...
                    'EdgeColor', 'none', 'FaceAlpha', 0.05);
            end
            % Plot the mean curve
            plot(ax, obj.Episodes, meanCurve, 'b-', 'LineWidth', 0.5,'Color',selectedColors(skillsBatch,:));

            % Plot knowledge plot for all skills
            % plot(ax, obj.Episodes, remainingKnowledge(remainingKnowledge(:,1)>obj.KnowledgeLowerBound,:), '-', 'LineWidth', 0.5,'Color',selectedColors(skillsBatch,:));

            % Plot dotted line for the KnowledgeLowerBound
            plot(ax, obj.FundamentalComplexity*ones(100,1), linspace(obj.KnowledgeLowerBound,1,100),'k--')
        end        
        
        %% Choose a set of skills to learn
        function [productSkills, skillClusterMembership] = getProductSkills(obj, NameValueArgs)
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.randStream
            end   
            if obj.REPEATING_SKILLS == 0
                % productSkills    = randperm(obj.TotalSkills,obj.MaxNumberOfSkillsPerProduct);

                useenSkills      = setdiff(1:obj.TotalSkills,obj.SeenSkills); % returns the data in A that is not in B, with no repetitions. C is in sorted order
                 
                if numel(useenSkills) >= obj.MaxNumberOfSkillsPerProduct
                    productSkillsIds = randperm(NameValueArgs.randStream, numel(useenSkills),obj.MaxNumberOfSkillsPerProduct);
                    productSkills    = useenSkills(productSkillsIds);
                else
                    numOfJokerSkills = obj.MaxNumberOfSkillsPerProduct - numel(useenSkills); 
                    %productSkills    = [useenSkills useenSkills(randperm(numel(useenSkills),numOfJokerSkills))];
                    productSkills    = [useenSkills useenSkills(randi(NameValueArgs.randStream, numel(useenSkills),1,numOfJokerSkills))];
                end

            else
                productSkills = randi(NameValueArgs.randStream, obj.TotalSkills,1,obj.MaxNumberOfSkillsPerProduct);
            end

            if obj.MaxNumberOfSkillsPerProduct<obj.NumberOfRobots
                extraSkills   = productSkills(randi(NameValueArgs.randStream, numel(productSkills),1,obj.NumberOfRobots-obj.MaxNumberOfSkillsPerProduct));
                productSkills = [productSkills, extraSkills];
            end

            % Track to which cluster the skill belongs to
            skillClusterMembership = zeros(size(productSkills));
            for s = 1:numel(productSkills)
                skillClusterMembership(s) =find(arrayfun(@(cluster) ismember(productSkills(s),obj.SkillClusters(:,cluster)),1:obj.TotalSkillClusters));
            end
        end

        %% Get number of learned skills from memory
        function getNumberOfLearnedSkillsPerAgent(obj)
            % Count the number of skills each agent has learned
            if obj.ENABLE_COLLECTIVE_LEARNING == 1
                % In CL all agents learned the new skills in the batch
                obj.NumberOfLearnedSkillsPerAgentAndTargetSkill  = numel(obj.SeenSkills).*ones(obj.NumberOfRobots,1);
            else
                for agent = 1:obj.NumberOfRobots
                    aux = obj.AgentStorage{agent,obj.SkillClusterMembership(agent)};
                    aux(isnan(aux)) = 0;
                    obj.NumberOfLearnedSkillsPerAgentAndTargetSkill(agent) = sum(unique(aux)~=0);
                end   
                obj.NumberOfLearnedSkillsPerAgentAndTargetSkill = reshape(obj.NumberOfLearnedSkillsPerAgentAndTargetSkill,obj.NumberOfRobots,1);
            end        
        end

        %% Log and update learned skills 
        function updateSkillPool(obj, NameValueArgs)
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.productSkills
                NameValueArgs.skillsBatch
                NameValueArgs.skillRemainingKnowledge
            end   
            productSkills = NameValueArgs.productSkills;
            skillsBatch   = NameValueArgs.skillsBatch;

            % Track to which cluster the skill belongs to
            skillClusterMembership = zeros(size(productSkills));
            for s = 1:numel(productSkills)
                skillClusterMembership(s) =find(arrayfun(@(cluster) ismember(productSkills(s),obj.SkillClusters(:,cluster)),1:obj.TotalSkillClusters));
            end

            % Check which skills were succesfully learned
            succesfullyLearnedProductSkills_id = NameValueArgs.skillRemainingKnowledge<=obj.KnowledgeLowerBound;
            
            % Replace the skill number by NaN if NOT learned
            productSkills(succesfullyLearnedProductSkills_id~=1) = NaN;

            % Add the learned skills in the batch to the skills each agent has individually learned
            obj.SkillsInAgent                   = cat(2,obj.SkillsInAgent,productSkills(:));%[obj.SkillsInAgent,productSkills'];

            % Update each agent's memory
            for agent = 1:obj.NumberOfRobots
                tmp                       = arrayfun(@(clstr) intersect(obj.SkillsInAgent(agent,:),obj.SkillClusters(:,clstr)),1:obj.TotalSkillClusters,'UniformOutput',false);
                obj.AgentMemory{agent}    = tmp;
                obj.ClusterTransferableKnowledgeFractionPerAgentAndTargetSkill(:,agent) = ...
                    arrayfun(@(clstr) numel(tmp{clstr}),1:obj.TotalSkillClusters)./obj.SkillsPerCluster;
                obj.AgentStorage(agent,:) = tmp;
            end

            % Check how many skills in the batch were new
            obj.NumberOfNewSkills(skillsBatch)  = numel(unique([obj.SeenSkills, productSkills(~isnan(productSkills))])) - numel(obj.SeenSkills); % The intersection of the sets
            
            % Log the indices of the NEW skills contained in the batch
            obj.SeenSkills                      = unique([obj.SeenSkills, productSkills(~isnan(productSkills))]); % The union of the sets
            
            % Log the number of NEW skills seen in the batch
            obj.NumberOfSeenSkills(skillsBatch) = numel(obj.SeenSkills);

            % Count the total number of skills learned from each cluster (by all agents)
            for k = 1:obj.TotalSkillClusters
                obj.SkillsInCluster(k,skillsBatch) = numel(intersect(obj.SeenSkills,obj.SkillClusters(:,k)));
            end
            obj.ClusterTransferableKnowledgeFraction = obj.SkillsInCluster(:,skillsBatch)./obj.SkillsPerCluster;
            disp("Cluster TOTAL transferable knowledge fraction:")
            disp(transpose(obj.ClusterTransferableKnowledgeFraction))
        end
        
        %% The dynamics of Isolated Learning (IsL)
        function out = isolatedLearningRemainingKnowledgeDynamics(obj)
            arguments
                obj (1,1) RobotCollective
            end            
            Alpha = diag(obj.Alpha_min + (obj.Alpha_max - obj.Alpha_min).*rand(obj.NumberOfRobots,1));
            out   = Alpha;
        end
        
        %% The dynamics of Incremental Learning (IL)
        function out = incrementalLearningRemainingKnowledgeDynamics(obj, NameValueArgs)
            arguments
                obj (1,1) RobotCollective
                NameValueArgs.eta
            end    
            out = (NameValueArgs.eta.*obj.NumberOfLearnedSkillsPerAgentAndTargetSkill.*double(obj.ENABLE_INCREMENTAL_LEARNING)) + 1;
        end

        %% The initial condition for Incremental Learning (IL)
        function out = incrementalLearningInitialRemainingKnowledge(obj) 
            arguments
                obj (1,1) RobotCollective
            end
            out = exp(-obj.Delta*obj.NumberOfLearnedSkillsPerAgentAndTargetSkill.*double(obj.ENABLE_INCREMENTAL_LEARNING));
        end

        %% Learning loop for a batch of skills
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

            if ~isempty(NameValueArgs.maxNumberOfProducts)
                % Use default value
                obj.MaxNumberOfProducts = NameValueArgs.maxNumberOfProducts;
            end      
        
            % Create figure to track progress =============================
            [fig, ax, selectedColors] = obj.generateCanvas();          
        
            % NOTE: Subindex jk means SKILL j in CLUSTER k
            complexity_jk_CL_Distributed = zeros(obj.NumberOfSkillBatches, 1);
            clusterKnowledge             = zeros(obj.NumberOfSkillBatches, 1);
            
            % Basics learning rates PER agent (embodiment dependent)
            % Alpha = mean([obj.Alpha_min, obj.Alpha_max])*eye(obj.NumberOfRobots);
            rng("default") % Reset the random number generator for consistency across simulations
            Alpha = obj.isolatedLearningRemainingKnowledgeDynamics();
            
            % Initialization 
            obj.NumberOfNewSkills  = zeros(1,obj.MaxNumberOfProducts);
            obj.NumberOfSeenSkills = zeros(1,obj.MaxNumberOfProducts);
            obj.SkillsInCluster    = zeros(obj.TotalSkillClusters,obj.MaxNumberOfProducts);
            
            % Loop over skill batches (products) ==========================
            rng("default") % Reset the random number generator for consistency across simulations
            
            % Rand stream for the products
            productRandomNumberStream = RandStream.create('mlfg6331_64');
            reset(productRandomNumberStream,0)

            for skillsBatch = 1:obj.NumberOfSkillBatches        
                % Select skills for a given product and determine their
                % cluster membership
                [productSkills, skillClusterMembership] = obj.getProductSkills(randStream = productRandomNumberStream);
                obj.SkillClusterMembership = skillClusterMembership;

                % Check wich skills in memory can be used to draw knowledge
                obj.getNumberOfLearnedSkillsPerAgent()
                disp("Skills in each agent's memory, given current skill's cluster")
                disp([productSkills; obj.SkillClusterMembership;obj.NumberOfLearnedSkillsPerAgentAndTargetSkill'])

                % Determine how much knowledge can be transfered from the
                % SOURCE clusters to the skill in the TARGET cluster
                % *NOTE: Use 0.99 instead of 1 to avoid eventual divisions by
                %        zero
                if obj.ENABLE_COLLECTIVE_LEARNING
                    aux = repmat(sum(obj.ClusterTransferableKnowledgeFractionPerAgentAndTargetSkill,2),1,obj.NumberOfRobots);
                    availableAggregatedScaledTransferableKnowledgetoClusters = min(0.99,obj.ClusterSimilarityMatrix*aux).*double(obj.ENABLE_TRANSFER_LEARNING);
                else
                    availableAggregatedScaledTransferableKnowledgetoClusters = min(0.99,obj.ClusterSimilarityMatrix*obj.ClusterTransferableKnowledgeFractionPerAgentAndTargetSkill).*double(obj.ENABLE_TRANSFER_LEARNING);
                end
                % Assign the transferable knowledge fraction according to the current cluster for each agent (depending on the cluster of the to-be-learned skills)
                obj.AggregatedScaledClusterTransferableKnowledgeFractionPerAgent(:) = diag(availableAggregatedScaledTransferableKnowledgetoClusters(skillClusterMembership,:));
                
                % For collective learning the cluster similarity matrix needs to be adapted
                % depending on the cluster of the skill being learned and the number of
                % agents
                for row = 1:obj.NumberOfRobots
                    for col = 1:obj.NumberOfRobots
                        if(skillClusterMembership(row) == skillClusterMembership(col))
                            obj.ClusterSimilarityMatrixPerAgent(row,col) = 1;
                        else
                            obj.ClusterSimilarityMatrixPerAgent(row,col) = obj.ClusterSimilarityMatrix(skillClusterMembership(row),skillClusterMembership(col));
                            
                        end
                    end
                end
            
                obj.ClusterTransferableKnowledgeFraction = zeros(obj.TotalSkillClusters,1);
                skillTransferableKnowledgeFraction = mean((obj.NumberOfLearnedSkillsPerAgentAndTargetSkill./obj.TotalSkills)*double(obj.ENABLE_TRANSFER_LEARNING));        
                clusterKnowledge(skillsBatch)      = mean(skillTransferableKnowledgeFraction);
                j      = skillsBatch;

                % fprintf('Robots = %2i | Eta_0 = %0.2f | Gamma_0 = %0.2f | Skills batch: %2i/%3i | Cluster knowledge: %1.3f \n',obj.NumberOfRobots, obj.Eta_0, obj.Gamma_0, skillsBatch, obj.NumberOfSkillBatches, skillTransferableKnowledgeFraction)
                cprintf('green','[INFO] Robots = %2i | Eta_0 = %0.2f | Gamma_0 = %0.2f | Skills batch: %2i/%3i | Cluster knowledge: %1.3f \n',obj.NumberOfRobots, obj.Eta_0, obj.Gamma_0, skillsBatch, obj.NumberOfSkillBatches, skillTransferableKnowledgeFraction)
                                
                % The initial conditions change when incremental
                % learning is active
                if obj.ENABLE_INCREMENTAL_LEARNING
                    initialRemainingKnowledge = diag(1 - obj.AggregatedScaledClusterTransferableKnowledgeFractionPerAgent)*(obj.incrementalLearningInitialRemainingKnowledge().*obj.SkillsRemainingKnowledge(productSkills));
                else
                    initialRemainingKnowledge = diag(1 - obj.AggregatedScaledClusterTransferableKnowledgeFractionPerAgent)*(obj.incrementalLearningInitialRemainingKnowledge().*ones(numel(productSkills),1));
                end

                % To speed up the simulation, once the KnowledgeLowerBound
                % is reached, stop the integration
                if  all(initialRemainingKnowledge <= obj.KnowledgeLowerBound)
                    remainingKnowledge = obj.KnowledgeLowerBound*ones(obj.NumberOfRobots,numel(obj.Episodes));
                else
                    remainingKnowledge = transpose( ...
                        ode4_sat(@(n,sigmaBar) ...
                            obj.collectiveKnowledgeSharingDynamicsEpisodic( ...
                                agentsRemainingKnowledge = sigmaBar, ...
                                isolatedLearningRemainingKnowledgeDynamics = Alpha), obj.Episodes, initialRemainingKnowledge)); 
                end

% remainingKnowledge = transpose( ...
%     ode4_sat(@(n,sigmaBar) ...
%         obj.collectiveKnowledgeSharingDynamicsEpisodic( ...
%             agentsRemainingKnowledge = sigmaBar, ...
%             isolatedLearningRemainingKnowledgeDynamics = Alpha), obj.Episodes, [initialRemainingKnowledge(1);zeros(7,1)])); 


                % Plot remaining knowlege curve
                obj.plotRemainingKnowledgeCurves(remainingKnowledge = remainingKnowledge, skillsBatch = skillsBatch, figureHandle=fig, axisHandle= ax,lineColors=selectedColors);
                
                % Determine the complexity for learning the skills in the batch
                % complexity_jk_CL_Distributed(j) = ...
                %     ceil( ...
                %         mean( ...
                %             obj.Episodes( ...
                %                 cell2mat( ...
                %                     arrayfun( ...
                %                         @(i) ( ...
                %                             find( ...
                %                                 remainingKnowledge(i,:)<obj.KnowledgeLowerBound,1,'first')),1:size(remainingKnowledge,1),'UniformOutput',false)))));
                % 


% Find skills in the batch that need to be learned
idx = find(initialRemainingKnowledge>obj.KnowledgeLowerBound);
if isempty(idx)
    complexity_jk_CL_Distributed(j) = 1;
else
complexity_jk_CL_Distributed(j) = ...
    ceil( ...
        mean( ...
            obj.Episodes( ...
                cell2mat( ...
                    arrayfun( ...
                        @(i) ( ...
                            find( ...
                                remainingKnowledge(i,:)<obj.KnowledgeLowerBound,1,'first')),idx,'UniformOutput',false)))));
end


                % For cases where a skill could not be learned, choose the
                % fundamental complexity as fallback value
                if isnan(complexity_jk_CL_Distributed(j))
                    complexity_jk_CL_Distributed(j) = obj.FundamentalComplexity;
                end                
                
                % Trigger warning if there are unlearned skills
                if any(remainingKnowledge(:,end)>obj.KnowledgeLowerBound)
                    numberNotLearnedSkills = numel(remainingKnowledge(remainingKnowledge(:,end)>obj.KnowledgeLowerBound));
                    warning('%3i skills COULD NOT be learned',numberNotLearnedSkills)
                end    
                        
                % Simulate the smart factory ==============================
                % productSkills = obj.getProductSkills();
                obj.SkillsRemainingKnowledge(productSkills) = remainingKnowledge(:,end);
                
                % Message if there were skills learned
                if any(obj.SkillsRemainingKnowledge(productSkills)<=obj.KnowledgeLowerBound)
                    disp('There are learned skill(s)!')
                end

                % Update the record of learned skills and their clusters
                obj.updateSkillPool(productSkills = productSkills, skillsBatch = skillsBatch, skillRemainingKnowledge = remainingKnowledge(:,end))
                % =========================================================================        
        
                % pause(1)
                % if obj.NumberOfLearnedSkillsPerAgentAndTargetSkill == obj.TotalSkills
                if sum(~isnan((unique(obj.SkillsInAgent(:))))) == obj.TotalSkills%numel(unique(obj.SkillsInAgent(:))) == obj.TotalSkills
                    warning('All skilles learned')
                    % Copy the last complexity to fill the remaining
                    % products
                    complexity_jk_CL_Distributed(j+1:end) = complexity_jk_CL_Distributed(j); 
                    pause(3)
                    break
                end
            end
              
            % Format plot
            fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 6, 1, 1)
            % pause(1)
            tightfig(fig);

            % Update the information about the learned skils
            obj.getNumberOfLearnedSkillsPerAgent()

            c_jk_cl_dist_episodes = (complexity_jk_CL_Distributed);
            learnedSkillsStorage  = obj.NumberOfLearnedSkillsPerAgentAndTargetSkill;
        
            % =============================================================            
            aux_fig = figure('Color', 'w');
            aux_ax = gca;
            pl1 = plot(1:obj.MaxNumberOfProducts, obj.NumberOfNewSkills,'r',LineWidt=1.5);
            hold on
            pl2 = plot(1:obj.MaxNumberOfProducts, obj.NumberOfSeenSkills,'b',LineWidt=1.5);
            clusterColors = distinguishable_colors(obj.TotalSkillClusters);
            pl3 = plot(1:obj.MaxNumberOfProducts, obj.SkillsInCluster,'-m',LineWidt=1.5);
            pl4 = plot(1:obj.MaxNumberOfProducts, obj.MaxNumberOfSkillsPerProduct*ones(size(1:obj.MaxNumberOfProducts)),'k--',LineWidt=1.5);
            colororder(clusterColors)
            % leg = legend([pl1 pl2 pl3],'No. new skills','No. learned skills','Skills per cluster');
            leg = legend('No. new skills','No. learned skills','Skills per cluster');
            xlabel('No. of products')
            ylabel('No. of learned skills')
            axis square
            xlim([1 skillsBatch])
            ylim([1 obj.TotalSkills])
            set(aux_ax, 'YScale', 'log')
            fcn_scrpt_prepare_graph_science_std(aux_fig, aux_ax, [pl1;pl2;pl3], [], [], 6, 1, 1);
        
            % Put together <results> structure
            results.c_jk_cl_dist_episodes = c_jk_cl_dist_episodes;
            results.learnedSkillsStorage  = learnedSkillsStorage;
            results.clusterKnowledge      = clusterKnowledge;
            results.numberOfNewSkills     = obj.NumberOfNewSkills;
            results.numberOfSeenSkills    = obj.NumberOfSeenSkills;
            results.SkillsInCluster       = obj.SkillsInCluster;
            results.skillsBatch           = skillsBatch;
        end

        %% Remaining knowledge dynamics functiion
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

            % Chane the incremental factor per episode
            eta_robots             = obj.Eta_0 + obj.Eta_std.*randn(obj.NumberOfRobots,1);    
            selfKnowledgeDynamics  = -isolatedLearningRemainingKnowledgeDynamics.*diag((1 - obj.AggregatedScaledClusterTransferableKnowledgeFractionPerAgent).^(-1)).*obj.incrementalLearningRemainingKnowledgeDynamics(eta = eta_robots).*eye(obj.NumberOfRobots);
        
        % numberOfRobotsPerCluster = numberOfRobots/obj.totalSkillClusters;
        % b = (clusterTrasferrableKnowledgeFraction-1)*((numberOfRobotsPerCluster-1) + (obj.totalSkillClusters-1)*numberOfRobotsPerCluster*clusterTrasferrableKnowledgeFraction);
        % a = Alpha(1,1);
        % fun = mean(f(eta_robots, NumberOfLearnedSkillsPerAgent));
        % % obj.gamma_0 =  (a/b)*fun + 2*abs((a/b)*fun);
        % disp([obj.gamma_0, a, b, (a/b), (a/b)*fun, NumberOfLearnedSkillsPerAgent, obj.gamma_0 > (a/b)*fun])
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
        % % clusterTransferMatrix = repmat(obj.AggregatedScaledClusterTransferableKnowledgeFractionPerAgent,obj.NumberOfRobots,1).*adjacencyMatrix;
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
            
            if obj.ENABLE_COLLECTIVE_LEARNING == 1 && obj.DISTRIBUTED_COLLECTIVE_LEARNING == 1
                % Remove the diagonal entries (corresponding to self-knowledge sharing)
                clusterTransferMatrix                    = obj.ClusterSimilarityMatrixPerAgent - eye(obj.NumberOfRobots);                       
                % Currupt randomly the values for sharing
                perturbation_B                           = rand(obj.NumberOfRobots,obj.NumberOfRobots);
                perturbation_B(clusterTransferMatrix==0) = 1;
                perturbation_B(clusterTransferMatrix==1) = 1;
                perturbation_B                           = (perturbation_B + transpose(perturbation_B))./2; % Ensure the matrix is symmetric
                
                clusterTransferMatrix                    = clusterTransferMatrix.*perturbation_B;
                
                % The result of inter-agent sharing scaled by the cluster
                % similarity matrix
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
        
        %% Knowledge integration function
        function distance = knowledgeIntegration(obj, neighborsKnowledge,agentKnowledge)
            alpha    = 2;
            % distance = exp(-alpha*abs(neighborsKnowledge-agentKnowledge));
            distance = exp(-alpha*(neighborsKnowledge-agentKnowledge).^2);
        end
              
        %% Gamma factor for iter agent knowledge sharing
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