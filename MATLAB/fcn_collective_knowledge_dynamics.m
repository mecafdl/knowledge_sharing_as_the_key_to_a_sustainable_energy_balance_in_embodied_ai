function d_coupledKnowledge = fcn_collective_knowledge_dynamics(agentsKnowledge, numberOfAgents, selfDynamics, adjacencyMatrix)
    integratedKnowledge  = zeros(numberOfAgents,1);
    for agent = 1:numberOfAgents
        % integratedKnowledge(agent) = ...
        %     adjacencyMatrix(agent,:)*(agentsKnowledge - agentsKnowledge(agent)*ones(numberOfAgents,1));
        % integratedKnowledge(agent) = ...
        %     adjacencyMatrix(agent,:)*(integrationFunction(agentsKnowledge,agentsKnowledge(agent)));
        integratedKnowledge(agent) = ...
            adjacencyMatrix(agent,:)*(fcn_knowledgeIntegration(agentsKnowledge,agentsKnowledge(agent)));
        % integratedKnowledge(agent) = ...
        %     sum(adjacencyMatrix(agent,:));
    end
    % integratedKnowledge
    degreeMatrix = diag(sum(adjacencyMatrix,2));
    laplacianMatrix = degreeMatrix - adjacencyMatrix; 
    aux = laplacianMatrix*agentsKnowledge;
    %d_coupledKnowledge = selfDynamics*agentsKnowledge - laplacianMatrix*agentsKnowledge;
    % d_coupledKnowledge = min(selfDynamics*agentsKnowledge - integratedKnowledge,0);
    % d_coupledKnowledge = selfDynamics*agentsKnowledge - integratedKnowledge;
    % d_coupledKnowledge = (selfDynamics - integratedKnowledge)*agentsKnowledge;
    % d_coupledKnowledge = (selfDynamics - integratedKnowledge)*agentsKnowledge;
    % d_coupledKnowledge = min(0,(diag(selfDynamics) - integratedKnowledge)).*agentsKnowledge;
    d_coupledKnowledge = (diag(selfDynamics) - integratedKnowledge).*agentsKnowledge;
    d_coupledKnowledge(agentsKnowledge<0) = 0;




    % degreeMatrix = diag(sum(adjacencyMatrix,2));
    % laplacianMatrix = degreeMatrix - adjacencyMatrix; 
    % d_coupledKnowledge = selfDynamics*agentsKnowledge - laplacianMatrix*agentsKnowledge;

end

function distance = fcn_knowledgeIntegration(neighborsKnowledge,agentKnowledge)
    alpha    = 2;
    distance = exp(-alpha*abs(neighborsKnowledge-agentKnowledge));
    distance = exp(-alpha*(neighborsKnowledge-agentKnowledge).^2);
end


% function out = integrationFunction(neighborsKnowledge,agentKnowledge)
%     % mean([agentsKnowledge, agentsKnowledge(agent)*ones(numberOfAgents,1)],2)
% 
%     distance = abs(neighborsKnowledge-agentKnowledge);
% 
%     out = exp(-3*distance);
% end