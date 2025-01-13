function [c_jk_cl_dist_episodes, learnedSkillsStorage, clusterKnowledge] = ...
    simulateDistributedCollectiveKnowledgeDynamicsGammaSweep(eta_0, gamma_0, parameters, numberOfRobots)
    g = @(delta, N_zeta) exp(-delta*N_zeta);    

    parameters.eta_0   = eta_0;
    parameters.gamma_0 = gamma_0;
    fig = figure('color','w');
    % Subindex jk means skill j cluster k
    % c_jk_cl_dist  = zeros(parameters.totalSkills/numberOfRobots, 1);
    if mod(parameters.totalSkills,numberOfRobots)==0
        numberOfSkillBatches = parameters.totalSkills/numberOfRobots;
        FLAG = 0;
    else
        numberOfSkillBatches = floor(parameters.totalSkills/numberOfRobots) + 1;
        FLAG = 1;
    end
    c_jk_cl_dist  = zeros(numberOfSkillBatches, 1);
    clusterKnowledge  = zeros(numberOfSkillBatches, 1);
    
    % New set of skills, new learning rates
    Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));
    learnedSkills = 0;
    for skillsBatch = 1:numberOfSkillBatches
        % if (skillsBatch==1)
        %     learnedSkills = 0;
        % else
        %     learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
        % end                
        % Beta adapts at every cycle
        % NOTE: From each cluster, only a fraction of knowlede can be transfered                
        % beta_k = numberOfRobots*learnedSkills/parameters.totalSkills;
        beta_k = learnedSkills/parameters.totalSkills;
        clusterKnowledge(skillsBatch) = beta_k;
        j      = skillsBatch;%learnedSkills + 1;
        fprintf('Gamma scenario: %2i | Eta scenario: %2i | Skills batch: %2i | Cluster knowledge: %1.3f \n',1, 1, skillsBatch, beta_k)
        % initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, numberOfRobots*learnedSkills).*ones(numberOfRobots,1);
        
        if FLAG == 0
            initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
            remainingKnowledge = transpose( ...
                ode4_sat(@(n,sigmaBar) ...
                    fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                    learnedSkills,...
                    sigmaBar, ...
                    numberOfRobots, ...
                    Alpha, ...
                    beta_k, ...
                    parameters, n), parameters.episodes, initialRemainingKnowedge));       

        elseif (FLAG == 1) 
            if (skillsBatch < numberOfSkillBatches)
                initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                remainingKnowledge = transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                        fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        learnedSkills,...
                        sigmaBar, ...
                        numberOfRobots, ...
                        Alpha, ...
                        beta_k, ...
                        parameters, n), parameters.episodes, initialRemainingKnowedge));       

            elseif (skillsBatch == numberOfSkillBatches)
                initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1),1);
                remainingKnowledge = transpose( ...
                    ode4_sat(@(n,sigmaBar) ...
                        fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                        learnedSkills,...
                        sigmaBar, ...
                        parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1), ...
                        Alpha(1:parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1),1:parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1)), ...
                        beta_k, ...
                        parameters, n), parameters.episodes, initialRemainingKnowedge));  
            end
        end

        learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
        if any(remainingKnowledge(:,end)==1)
            warning('There are unlearned skills')
        end                

        c_jk_cl_dist(j) = ...
            ceil( ...
                mean( ...
                    parameters.episodes( ...
                        cell2mat( ...
                            arrayfun( ...
                                @(i) min( ...
                                    find( ...
                                        remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:size(remainingKnowledge,1),'UniformOutput',false)))));
        
        if isnan(c_jk_cl_dist(j))
            % disp('here')
            c_jk_cl_dist(j) = parameters.fundamentalComplexity;
        end


        loglog(parameters.episodes,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
        xlim([0 parameters.episodes(end)])
        ylim([0.008 1E0])
        xticks([10^0 10^1 10^2])
        xticklabels({'1','10', '$c_0$'})
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
    c_jk_cl_dist_episodes = sum(c_jk_cl_dist);
    learnedSkillsStorage  = learnedSkills;

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