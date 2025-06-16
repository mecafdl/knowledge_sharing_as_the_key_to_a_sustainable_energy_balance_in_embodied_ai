function d_coupledRemainingKnowledge = ...
    fcn_collectiveKnowledgeSharingDynamicsEpisodicGammaSweep( ...
        numberOfLearnedSkills, ...
        agentsRemainingKnowledge, ...
        numberOfRobots, ...
        Alpha, ...
        clusterTrasferrableKnowledgeFraction, ...
        parameters, ...
        episodeIntegrationStep)
   

%     gammaValues = linspace(-0.2,2,1000);
% % linspace(-0.2,2,numel(parameters.episodes));
%     gammaValues = [gammaValues, gammaValues(end)*ones(1,4001)];

    gammaValues = 0.022*parameters.episodes - 0.2;

    % eta_robots             = pearsrnd(eta_0, eta_factor*eta_0, 1, 10, numberOfRobots,1);
    % eta_robots             = pearsrnd(parameters.eta_0, parameters.eta_std, 1, 10, numberOfRobots,1);
    % eta_robots             = parameters.eta_0 + parameters.eta_std.*randn(numberOfRobots,1);
    
    
    eta_robots             =  parameters.eta_0 + parameters.eta_std.*randn(numberOfRobots,1);
    % eta_robots             =  eta_test(episodeIntegrationStep*10 + 1) + parameters.eta_std.*randn(numberOfRobots,1);
    
    
    selfKnowledgeDynamics  = -Alpha.*((1 - clusterTrasferrableKnowledgeFraction).^(-1)).*f(eta_robots, numberOfLearnedSkills).*eye(numberOfRobots);


numberOfRobotsPerCluster = numberOfRobots/parameters.totalSkillClusters;
b = (clusterTrasferrableKnowledgeFraction-1)*((numberOfRobotsPerCluster-1) + (parameters.totalSkillClusters-1)*numberOfRobotsPerCluster*clusterTrasferrableKnowledgeFraction);
a = Alpha(1,1);
fun = mean(f(eta_robots, numberOfLearnedSkills));
% parameters.gamma_0 =  (a/b)*fun + 2*abs((a/b)*fun);
disp([parameters.gamma_0, a, b, (a/b), (a/b)*fun, numberOfLearnedSkills, parameters.gamma_0 > (a/b)*fun])

if ~(parameters.gamma_0 > (a/b)*fun)
    warning('Condition violated')
end



    integratedKnowledgeDynamics  = zeros(numberOfRobots,1);
    weightedAdjacencyMatrix      = fcn_generateGamma(parameters.gamma_0, parameters.gamma_std, numberOfRobots, parameters.randCommInterrupt);
    
% gamma_index = ceil(episodeIntegrationStep*10 + 1);
% weightedAdjacencyMatrix      = fcn_generateGamma(gammaValues(gamma_index), parameters.gamma_std, numberOfRobots);

    % If collective learning is distributed, the <weightedAdjacencyMatrix>
    % needs to be scaled down by the cluster knowledge fraction transfer
    % factor beta_k
    if parameters.cl_distributed == 1
        adjacencyMatrix       = double(weightedAdjacencyMatrix~=0);
        clusterTransferMatrix = clusterTrasferrableKnowledgeFraction*adjacencyMatrix;
        for i = 1:parameters.totalSkillClusters
           clusterTransferMatrix((numberOfRobots/parameters.totalSkillClusters)*(i-1) + 1:(numberOfRobots/parameters.totalSkillClusters)*i,(numberOfRobots/parameters.totalSkillClusters)*(i-1) + 1:(numberOfRobots/parameters.totalSkillClusters)*i) = ones(numberOfRobots/parameters.totalSkillClusters) - eye(numberOfRobots/parameters.totalSkillClusters);
        end
        clusterTransferMatrix = triu(clusterTransferMatrix) + transpose(triu(clusterTransferMatrix));
        perturbation_B = rand(numberOfRobots,numberOfRobots);
        perturbation_B(clusterTransferMatrix==0) = 1;
        perturbation_B(clusterTransferMatrix==1) = 1;
        % Ensure the matrix is symmetric
        perturbation_B   = (perturbation_B + transpose(perturbation_B))./2;
        
perturbation_B = ones(numberOfRobots);
        
        clusterTransferMatrix = clusterTransferMatrix.*perturbation_B;
        if sum(clusterTransferMatrix-clusterTransferMatrix','all') ~= 0
            warning('Error in B matrix')
        end
        weightedAdjacencyMatrix = clusterTransferMatrix.*weightedAdjacencyMatrix;
    end
    
    % In agents are not sharing knowledge, cancel the connectivity matrix
    if parameters.enableSharing == 0
        weightedAdjacencyMatrix = 0*weightedAdjacencyMatrix;
    end

    % =====================================================================

    for agent = 1:numberOfRobots
        %knowledgeIntegrationVector         = fcn_knowledgeIntegration(agentsRemainingKnowledge,agentsRemainingKnowledge(agent));
        knowledgeIntegrationVector         = ones(numberOfRobots,1);
        % if any(knowledgeIntegrationVector<0.99)
        %     disp('HERE')
        % end
        % pause(0.5)
        integratedKnowledgeDynamics(agent) = ...
            weightedAdjacencyMatrix(agent,:)*knowledgeIntegrationVector;
    end
% % weightedAdjacencyMatrix
% % TheMatrix = Alpha.*((1 - clusterTrasferrableKnowledgeFraction).^(-1)).*f(eta_robots, numberOfLearnedSkills).*eye(numberOfRobots) + weightedAdjacencyMatrix;
% eigenvalues = eig(-TheMatrix);
% 
% if all(eigenvalues>0)
%     disp('HERE')
% end
% % disp([gammaValues(gamma_index) eigenvalues'])
% disp([parameters.gamma_0 eigenvalues'])


% numberOfRobotsPerCluster = numberOfRobots/parameters.totalSkillClusters;

% tmp1 = (clusterTrasferrableKnowledgeFraction-1)*((numberOfRobotsPerCluster-1) + (parameters.totalSkillClusters-1)*numberOfRobotsPerCluster*clusterTrasferrableKnowledgeFraction)

% collectiveTerm = parameters.gamma_0*(clusterTrasferrableKnowledgeFraction-1)*((numberOfRobotsPerCluster-1) + (parameters.totalSkillClusters-1)*numberOfRobotsPerCluster*clusterTrasferrableKnowledgeFraction);
% individualTem = Alpha*f(eta_robots, numberOfLearnedSkills);
% [individualTem(1), collectiveTerm, individualTem(1)>collectiveTerm]

% 
% b = (clusterTrasferrableKnowledgeFraction-1)*((numberOfRobotsPerCluster-1) + (parameters.totalSkillClusters-1)*numberOfRobotsPerCluster*clusterTrasferrableKnowledgeFraction);
% a = Alpha(1,1);
% fun = mean(f(eta_robots, numberOfLearnedSkills));
% disp([parameters.gamma_0, (a/b)*fun, parameters.gamma_0 > (a/b)*fun numberOfLearnedSkills])
% 
% if ~(parameters.gamma_0 > (a/b)*fun)
%     warning('Condition violated')
% end




    mainDynamicsCoeff  = (diag(selfKnowledgeDynamics) - integratedKnowledgeDynamics);
    if any(mainDynamicsCoeff > 0)
        disp('Knowledge corruption!')
    end
    d_coupledRemainingKnowledge = mainDynamicsCoeff.*agentsRemainingKnowledge;
    % d_coupledRemainingKnowledge = -TheMatrix*agentsRemainingKnowledge
    
    % if any(abs(d_coupledRemainingKnowledge_1 - d_coupledRemainingKnowledge)>1E-6)
    %     bbb= 0;
    % end
    
    % Constraints to stop the knowledge evolution when the minimum
    % threshold is reached
    d_coupledRemainingKnowledge(agentsRemainingKnowledge<parameters.knowledgeLowerBound) = 0;
    % d_coupledRemainingKnowledge(agentsRemainingKnowledge>1) = 0;
end

function distance = fcn_knowledgeIntegration(neighborsKnowledge,agentKnowledge)
    alpha    = 2;
    % distance = exp(-alpha*abs(neighborsKnowledge-agentKnowledge));
    distance = exp(-alpha*(neighborsKnowledge-agentKnowledge).^2);
end

function out = f(eta, N_zeta)
    out = eta.*N_zeta+1;
end


% function d_coupledRemainingKnowledge = ...
%     fcn_collectiveKnowledgeSharingDynamicsEpisodicGammaSweep( ...
%         numberOfLearnedSkills, ...
%         agentsRemainingKnowledge, ...
%         numberOfRobots, ...
%         Alpha, ...
%         clusterTrasferrableKnowledgeFraction, ...
%         parameters, ...
%         episodeIntegrationStep)
% 
% 
% %     gammaValues = linspace(-0.2,2,1000);
% % % linspace(-0.2,2,numel(parameters.episodes));
% %     gammaValues = [gammaValues, gammaValues(end)*ones(1,4001)];
% 
%     gammaValues = 0.022*parameters.episodes - 0.2;
% 
%     % eta_robots             = pearsrnd(eta_0, eta_factor*eta_0, 1, 10, numberOfRobots,1);
%     % eta_robots             = pearsrnd(parameters.eta_0, parameters.eta_std, 1, 10, numberOfRobots,1);
%     % eta_robots             = parameters.eta_0 + parameters.eta_std.*randn(numberOfRobots,1);
% 
% 
%     eta_robots             =  parameters.eta_0 + parameters.eta_std.*randn(numberOfRobots,1);
%     % eta_robots             =  eta_test(episodeIntegrationStep*10 + 1) + parameters.eta_std.*randn(numberOfRobots,1);
% 
% 
%     selfKnowledgeDynamics  = -Alpha.*((1 - clusterTrasferrableKnowledgeFraction).^(-1)).*f(eta_robots, numberOfLearnedSkills).*eye(numberOfRobots);
% 
% 
%     integratedKnowledgeDynamics  = zeros(numberOfRobots,1);
%     weightedAdjacencyMatrix      = fcn_generateGamma(parameters.gamma_0, parameters.gamma_std, numberOfRobots, parameters.randCommInterrupt);
% 
% % gamma_index = ceil(episodeIntegrationStep*10 + 1);
% % weightedAdjacencyMatrix      = fcn_generateGamma(gammaValues(gamma_index), parameters.gamma_std, numberOfRobots);
% 
%     % If collective learning is distributed, the <weightedAdjacencyMatrix>
%     % needs to be scaled down by the cluster knowledge fraction transfer
%     % factor beta_k
%     if parameters.cl_distributed == 1
%         adjacencyMatrix       = double(weightedAdjacencyMatrix~=0);
%         clusterTransferMatrix = clusterTrasferrableKnowledgeFraction*adjacencyMatrix;
%         for i = 1:parameters.totalSkillClusters
%            clusterTransferMatrix((numberOfRobots/parameters.totalSkillClusters)*(i-1) + 1:(numberOfRobots/parameters.totalSkillClusters)*i,(numberOfRobots/parameters.totalSkillClusters)*(i-1) + 1:(numberOfRobots/parameters.totalSkillClusters)*i) = ones(numberOfRobots/parameters.totalSkillClusters) - eye(numberOfRobots/parameters.totalSkillClusters);
%         end
%         clusterTransferMatrix = triu(clusterTransferMatrix) + transpose(triu(clusterTransferMatrix));
%         perturbation_B = rand(numberOfRobots,numberOfRobots);
%         perturbation_B(clusterTransferMatrix==0) = 1;
%         perturbation_B(clusterTransferMatrix==1) = 1;
%         % Ensure the matrix is symmetric
%         perturbation_B   = (perturbation_B + transpose(perturbation_B))./2;
% 
% perturbation_B = ones(numberOfRobots);
% 
%         clusterTransferMatrix = clusterTransferMatrix.*perturbation_B
%         if sum(clusterTransferMatrix-clusterTransferMatrix','all') ~= 0
%             warning('Error in B matrix')
%         end
%         weightedAdjacencyMatrix = clusterTransferMatrix.*weightedAdjacencyMatrix;
%     end
% 
%     % In agents are not sharing knowledge, cancel the connectivity matrix
%     if parameters.enableSharing == 0
%         weightedAdjacencyMatrix = 0*weightedAdjacencyMatrix;
%     end
% 
%     % =====================================================================
% 
%     for agent = 1:numberOfRobots
%         %knowledgeIntegrationVector         = fcn_knowledgeIntegration(agentsRemainingKnowledge,agentsRemainingKnowledge(agent));
%         knowledgeIntegrationVector         = ones(numberOfRobots,1);
%         % if any(knowledgeIntegrationVector<0.99)
%         %     disp('HERE')
%         % end
%         % pause(0.5)
%         integratedKnowledgeDynamics(agent) = ...
%             weightedAdjacencyMatrix(agent,:)*knowledgeIntegrationVector;
%     end
% % weightedAdjacencyMatrix
% TheMatrix = Alpha.*((1 - clusterTrasferrableKnowledgeFraction).^(-1)).*f(eta_robots, numberOfLearnedSkills).*eye(numberOfRobots) + weightedAdjacencyMatrix;
% 
% 
% eigenvalues = eig(-TheMatrix);
% 
% if all(eigenvalues>0)
%     disp('HERE')
% end
% 
% % disp([gammaValues(gamma_index) eigenvalues'])
% 
% disp([parameters.gamma_0 eigenvalues'])
% 
% 
%     mainDynamicsCoeff  = (diag(selfKnowledgeDynamics) - integratedKnowledgeDynamics);
%     % if any(coeff>0)
%     %     disp('Knowledge corruption!')
%     % end
%     d_coupledRemainingKnowledge_1 = mainDynamicsCoeff.*agentsRemainingKnowledge
%     d_coupledRemainingKnowledge = -TheMatrix*agentsRemainingKnowledge
% 
%     if any(abs(d_coupledRemainingKnowledge_1 - d_coupledRemainingKnowledge)>1E-6)
%         bbb= 0;
%     end
% 
%     % Constraints to stop the knowledge evolution when the minimum
%     % threshold is reached
%     % d_coupledRemainingKnowledge(agentsRemainingKnowledge<parameters.knowledgeLowerBound) = 0;
%     % d_coupledRemainingKnowledge(agentsRemainingKnowledge>1) = 0;
% end
% 
% function distance = fcn_knowledgeIntegration(neighborsKnowledge,agentKnowledge)
%     alpha    = 2;
%     % distance = exp(-alpha*abs(neighborsKnowledge-agentKnowledge));
%     distance = exp(-alpha*(neighborsKnowledge-agentKnowledge).^2);
% end
% 
% function out = f(eta, N_zeta)
%     out = eta.*N_zeta+1;
% end