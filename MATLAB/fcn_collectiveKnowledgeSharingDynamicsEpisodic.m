function d_coupledRemainingKnowledge = ...
    fcn_collectiveKnowledgeSharingDynamicsEpisodic( ...
        numberOfLearnedSkills, ...
        agentsRemainingKnowledge, ...
        numberOfRobots, ...
        Alpha, ...
        beta_k, ...
        parameters)

    % fcn_collectiveKnowledgeSharingDynamics( ...
    %     agentsRemainingKnowledge, ...
    %     numberOfRobots, ...
    %     selfKnowledgeDynamics, ...
    %     clusterTransferMatrix, ...
    %     parameters)
 

    % eta_robots             = pearsrnd(eta_0, eta_factor*eta_0, 1, 10, numberOfRobots,1);
    % eta_robots             = pearsrnd(parameters.eta_0, parameters.eta_std, 1, 10, numberOfRobots,1);
    % eta_robots             = parameters.eta_0 + parameters.eta_std.*randn(numberOfRobots,1);
    eta_robots             =  parameters.eta_0 + parameters.eta_std.*randn(numberOfRobots,1);
    selfKnowledgeDynamics  = -Alpha.*((1 - beta_k).^(-1)).*f(eta_robots, numberOfLearnedSkills).*eye(numberOfRobots);


    integratedKnowledgeDynamics  = zeros(numberOfRobots,1);
    weightedAdjacencyMatrix      = fcn_generateGamma(parameters.gamma_0, parameters.gamma_std, numberOfRobots);

    if parameters.cl_distributed == 1
        adjacencyMatrix       = double(weightedAdjacencyMatrix~=0);
        clusterTransferMatrix = beta_k*adjacencyMatrix;
        for i=1:parameters.totalSkillClusters
           clusterTransferMatrix((numberOfRobots/parameters.totalSkillClusters)*(i-1) + 1:(numberOfRobots/parameters.totalSkillClusters)*i,(numberOfRobots/parameters.totalSkillClusters)*(i-1) + 1:(numberOfRobots/parameters.totalSkillClusters)*i) = ones(numberOfRobots/parameters.totalSkillClusters) - eye(numberOfRobots/parameters.totalSkillClusters);
        end
        clusterTransferMatrix = triu(clusterTransferMatrix) + transpose(triu(clusterTransferMatrix));
        perturbation_B = rand(numberOfRobots,numberOfRobots);
        perturbation_B(clusterTransferMatrix==0) = 1;
        perturbation_B(clusterTransferMatrix==1) = 1;
        % Ensure the matrix is symmetric
        perturbation_B   = (perturbation_B + transpose(perturbation_B))./2;
        clusterTransferMatrix = clusterTransferMatrix.*perturbation_B;
        if sum(clusterTransferMatrix-clusterTransferMatrix','all') ~= 0
            warning('Error in B matrix')
        end
        weightedAdjacencyMatrix = clusterTransferMatrix.*weightedAdjacencyMatrix;
    end

    if parameters.enableSharing == 0
        weightedAdjacencyMatrix = 0*weightedAdjacencyMatrix;
    end

    % =====================================================================

    for agent = 1:numberOfRobots
        integratedKnowledgeDynamics(agent) = ...
            weightedAdjacencyMatrix(agent,:)*(fcn_knowledgeIntegration(agentsRemainingKnowledge,agentsRemainingKnowledge(agent)));
    end
    coeff  = (diag(selfKnowledgeDynamics) - integratedKnowledgeDynamics);
    % if any(coeff>0)
    %     disp('Knowledge corruption!')
    % end
    d_coupledRemainingKnowledge = (diag(selfKnowledgeDynamics) - integratedKnowledgeDynamics).*agentsRemainingKnowledge;
    % Constraints to keep the remaining knowledge  in [0,1]
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