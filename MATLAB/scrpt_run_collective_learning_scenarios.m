clearvars
clc
close all
cd(fileparts(matlab.desktop.editor.getActiveFilename));

%%
close all
clc

theEtas   = [-0.1 0 0.1];
theGammas = [-0.2 0 0.2];
[X, Y]    = meshgrid(theEtas,theGammas);
scenarios = [X(:) Y(:)];
numberOfScenarios = size(scenarios,1);
   
% robotCollectivesSize = [4,8,16,32,64,128];
robotCollectivesSize = 4:4:128;
cl_scenarios_results = cell(numberOfScenarios,numel(robotCollectivesSize));

tStart = tic;
for scenarioIndex = 1:numberOfScenarios
    for robotCollectiveIndex = 1:numel(robotCollectivesSize)  
        close all
        % Initialize a robot collective
        theCollective = RobotCollective(numberOfRobots  = robotCollectivesSize(robotCollectiveIndex), ...
                                        repeatingSkills = false);
        theCollective.RAND_COMM_INTERRUPT         = true; 
        theCollective.ENABLE_INCREMENTAL_LEARNING = true; 
        theCollective.ENABLE_TRANSFER_LEARNING    = true; 
        theCollective.ENABLE_COLLECTIVE_LEARNING  = true; 
        
        % Define simulation episodes
        theCollective.Episodes            = 1:0.1:2*theCollective.FundamentalComplexity;
        
        % Define agent(s) learning factors
        theCollective.Eta_0   = scenarios(scenarioIndex,1);
        theCollective.Gamma_0 = scenarios(scenarioIndex,2);    
        
        % Max. nummber of skills per product
        theCollective.MaxNumberOfSkillsPerProduct = robotCollectivesSize(robotCollectiveIndex);

        % Run the scenario
        cprintf('green',sprintf('[INFO] DCL: collective size %2i | eta_0 = %0.2f | gamma_0 = %0.2f ...\n', robotCollectivesSize(robotCollectiveIndex),theCollective.Eta_0,theCollective.Gamma_0))
        pause(1)
        [~, ~, ~, cl_scenarios_results{scenarioIndex, robotCollectiveIndex}] = ...
        theCollective.simulateDistributedCollectiveKnowledgeDynamics(maxNumberOfProducts = 1000);
    end
end
tEnd = toc(tStart);