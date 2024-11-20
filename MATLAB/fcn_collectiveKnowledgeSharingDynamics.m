function d_coupledRemainingKnowledge = ...
    fcn_collectiveKnowledgeSharingDynamics( ...
        agentsRemainingKnowledge, ...
        numberOfAgents, ...
        selfKnowledgeDynamics, ...
        clusterTransferMatrix, ...
        parameters)
 
    integratedKnowledgeDynamics  = zeros(numberOfAgents,1);

    % =====================================================================
    % * NOTE: If the flag EPISODIC is true then the weights gamma of the
    %         adjacency matrix will be sampled according to a Pearson
    %         distribution at every step.
    weightedAdjacencyMatrix = fcn_generateGamma(parameters.gamma_0, parameters.gamma_std, numberOfAgents);
    weightedAdjacencyMatrix = clusterTransferMatrix.*weightedAdjacencyMatrix;
    % =====================================================================

    for agent = 1:numberOfAgents
        integratedKnowledgeDynamics(agent) = ...
            weightedAdjacencyMatrix(agent,:)*(fcn_knowledgeIntegration(agentsRemainingKnowledge,agentsRemainingKnowledge(agent)));
    end
    coeff  = (diag(selfKnowledgeDynamics) - integratedKnowledgeDynamics);
    if any(coeff>0)
        disp('Knowledge corruption!')
    end
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