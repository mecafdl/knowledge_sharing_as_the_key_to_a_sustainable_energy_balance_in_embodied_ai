function plotSkillClusters2(theCollective)
    close all
%     N           = theCollective.TotalSkills;          % total points
%     K           = theCollective.TotalSkillClusters;            % clusters
%     ptsPerClust = N/K;
% 
%     rng('default');
%     centers = [-5, -5; -5, 5; 5, -5; 5, 5];  % 4 cluster centers
% 
%     X = zeros(N,2);
%     for k = 1:K
%         idx = (1:ptsPerClust) + (k-1)*ptsPerClust;
%         X(idx,:) = centers(k,:) + randn(ptsPerClust,2);
%     end
% 
%     % Visualize
%     figure(color="w");
%     sp = scatter(X(:,1), X(:,2), 30, repelem(1:K, ptsPerClust)', 'filled',MarkerFaceAlpha=0.2);
%     axis square;
%     title('Skill Clusters');
%     axis off
%     hold on
% 
%     for pnt = 1:size(theCollective.SkillsInAgent,2)
%         scatter(X(theCollective.SkillsInAgent(:,pnt),1), X(theCollective.SkillsInAgent(:,pnt),2), 30, sp.CData(theCollective.SkillsInAgent(:,pnt)), 'filled',MarkerEdgeColor='k',MarkerFaceAlpha=0.5);
%         h = scatter(X(theCollective.SkillsInAgent(:,pnt),1), X(theCollective.SkillsInAgent(:,pnt),2), 50, sp.CData(theCollective.SkillsInAgent(:,pnt)),MarkerEdgeColor='r',LineWidth=2,MarkerFaceAlpha=1);
%         pause(0.5)
%         delete(h)
%     end
% 
% end
% 

% MATLAB Script to Plot N 2D Points Separated into M Clusters

% Define the number of points per cluster
pointsPerCluster = theCollective.SkillsPerCluster; % Number of 2D points in EACH cluster

% The similarity matrix S(i,j) indicates how similar cluster i is to cluster j.
% Higher values mean more similarity.
S = theCollective.ClusterSimilarityMatrix + eye(theCollective.TotalSkillClusters);

% Ensure the similarity matrix is square
if size(S, 1) ~= size(S, 2)
    error('The similarity matrix must be square.');
end

% The number of clusters (m) is now determined by the size of the similarity matrix
numberOfClusters = theCollective.TotalSkillClusters;

% Convert the similarity matrix to a dissimilarity (distance) matrix.
% Common methods:
% 1. dissimilarity = 1 - similarity (for similarities in [0,1])
% 2. dissimilarity = max(S(:)) - S
% Let's use the first method, assuming similarities are normalized between 0 and 1.
D = 1 - S;

% Perform Multi-Dimensional Scaling (MDS) to get 2D coordinates for cluster centers.
% cmdscale (classical MDS) is suitable for converting a distance matrix into coordinates.
% We request 2 dimensions for 2D plotting.
clusterCenters = cmdscale(D, 2);

% Calculate the total number of points based on the inputs
totalSkills = numberOfClusters * pointsPerCluster; % Total number of 2D points

% --- 2. Generate N 2D Points and Assign to Clusters ---
% Each point will be generated around one of the cluster centers with some noise.
points = zeros(totalSkills, 2);   % Preallocate matrix for points (x, y)
clusterIDs = zeros(totalSkills, 1); % Preallocate vector for cluster assignments

currentPointIdx = 1;

for i = 1:numberOfClusters
    % The number of points for the current cluster is a direct input
    numPointsInCurrentCluster = pointsPerCluster;

    % Generate points around the current cluster center
    % Add some random noise (e.g., standard deviation of 0.5)
    % Reduce noise stddev since cluster centers are now derived from distances
    noise = 0.05 * randn(numPointsInCurrentCluster, 2);
    
    % Assign points
    points(currentPointIdx : currentPointIdx + numPointsInCurrentCluster - 1, :) = ...
        repmat(clusterCenters(i, :), numPointsInCurrentCluster, 1) + noise;
    
    % Assign cluster ID
    clusterIDs(currentPointIdx : currentPointIdx + numPointsInCurrentCluster - 1) = i;
    
    currentPointIdx = currentPointIdx + numPointsInCurrentCluster;
end

% --- 3. Plot the Points, Colored by Cluster ---
% Use gscatter for convenient plotting of grouped data.
% It automatically assigns different colors to different groups.

figure(Color='w'); % Create a new figure window

% Define a set of distinct colors for each cluster
% You can customize these RGB triplets or use built-in colormaps (e.g., hsv(m), parula(m))
clusterColors = [
    0.8500 0.3250 0.0980; % Orange
    0.0000 0.4470 0.7410; % Blue
    0.9290 0.6940 0.1250; % Yellow
    0.4940 0.1840 0.5560; % Purple
    0.4660 0.6740 0.1880; % Green
    0.3010 0.7450 0.9330; % Light Blue
    0.6350 0.0780 0.1840; % Red-Brown
];

% Ensure we have enough colors for 'm' clusters. If not, generate more.
if numberOfClusters > size(clusterColors, 1)
    % Generate more distinct colors if 'm' exceeds the predefined list
    % This requires the 'distinguishable_colors' function from MATLAB File Exchange,
    % or you can use a built-in colormap like `clusterColors = hsv(m);`
    % For this example, we'll use hsv(m) for simplicity if distinguishable_colors isn't available.
    clusterColors = hsv(numberOfClusters); 
end

% --- Plotting the shaded areas (alpha shapes) first for smooth borders ---
% This ensures the scatter points and centers are visible on top
% 'alpha' parameter controls the "tightness" of the boundary.
% A smaller alpha value creates a tighter boundary; a larger value creates a looser one.
% Adjust this value based on your data's spread.
alpha_value = 5.0; 

for i = 1:numberOfClusters
    % Get points belonging to the current cluster
    clusterPoints = points(clusterIDs == i, :);
    
    % Check if there are enough points to form an alpha shape
    if size(clusterPoints, 1) >= 3
        % Create an alphaShape object
        shp = alphaShape(clusterPoints(:,1), clusterPoints(:,2), alpha_value);
        
        % Plot the alpha shape
        plot(shp, 'FaceColor', clusterColors(i,:), ...
                 'FaceAlpha', 0.1, ...
                 'EdgeColor', 'none', ...
                 'LineWidth', 0.5);
    end
    hold on
end


hold on
% gscatter(X, Y, G, colors, markers, sizes, 'on', xlabel, ylabel)
% The 'colors' argument is now used to specify the colors for each group.
h = gscatter(points(:,1), points(:,2), clusterIDs, clusterColors(1:numberOfClusters,:), 'o', 5, 'filled','MarkerFaceAlpha',0.2);

% Customize plot appearance
title(sprintf('Scatter Plot of %d Points in %d Clusters (%d points/cluster) from Similarity Matrix', totalSkills, numberOfClusters, pointsPerCluster));
% xlabel('Dimension 1 (from MDS)');
% ylabel('Dimension 2 (from MDS)');
grid on;
axis square
axis off; % Ensures that the aspect ratio is equal for x and y axes

% Customize marker edge colors for each cluster
% Here, we explicitly set all marker edges to black for better contrast.
for i = 1:numberOfClusters
    % h(i) refers to the scatter object for the i-th group
    h(i).MarkerEdgeColor = [0 0 0]; % Black edge
    h(i).LineWidth = 1; % Adjust line width for visibility
end

% Optional: Display cluster centers (larger markers)
hold on; % Keep the current plot
% Adjust center marker size and style for derived centers
% scatter(clusterCenters(:,1), clusterCenters(:,2), 250, 'k', 'pentagram', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 2);
% text(clusterCenters(:,1) + 0.1, clusterCenters(:,2) + 0.1, string(1:numberOfClusters)', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');



    % for pnt = 1:size(theCollective.SkillsInAgent,2)
    %     scatter(points(theCollective.SkillsInAgent(:,pnt),1), points(theCollective.SkillsInAgent(:,pnt),2), 30, clusterColors(clusterIDs(theCollective.SkillsInAgent(:,pnt)),:), 'filled',MarkerEdgeColor='k',MarkerFaceAlpha=0.5);
    %     h2 = scatter(points(theCollective.SkillsInAgent(:,pnt),1), points(theCollective.SkillsInAgent(:,pnt),2), 30, clusterColors(clusterIDs(theCollective.SkillsInAgent(:,pnt)),:),MarkerEdgeColor='r',LineWidth=2);
    % 
    %     % h2 = scatter(points(theCollective.SkillsInAgent(:,pnt),1), points(theCollective.SkillsInAgent(:,pnt),2), 50, sp.CData(theCollective.SkillsInAgent(:,pnt)),MarkerEdgeColor='r',LineWidth=2,MarkerFaceAlpha=1);
    %     pause(0.5)
    %     delete(h2)
    % end

    agentColors= distinguishable_colors(theCollective.NumberOfRobots);
    for pnt = 1:size(theCollective.SkillsInAgent,2)
        scatter(points(theCollective.SkillsInAgent(:,pnt),1), points(theCollective.SkillsInAgent(:,pnt),2), 30, agentColors, 'filled',MarkerEdgeColor='k',MarkerFaceAlpha=0.5);
        h2 = scatter(points(theCollective.SkillsInAgent(:,pnt),1), points(theCollective.SkillsInAgent(:,pnt),2), 30, agentColors,MarkerEdgeColor='r',LineWidth=2);

        % h2 = scatter(points(theCollective.SkillsInAgent(:,pnt),1), points(theCollective.SkillsInAgent(:,pnt),2), 50, sp.CData(theCollective.SkillsInAgent(:,pnt)),MarkerEdgeColor='r',LineWidth=2,MarkerFaceAlpha=1);
        pause(0.5)
        delete(h2)
    end

hold off;

% legend('Location', 'bestoutside'); % Adjust legend position





% N = 200;
% X = rand(N,2)*20 - 10;   % random initial positions
% 
% K = 4;
% [idx, C] = kmeans(X, K, 'Replicates',5);
% 
% % Reassign each point to its cluster centroid
% X2 = C(idx, :);
% 
% figure;
% gscatter(X2(:,1), X2(:,2), idx, 'rgbm', 'o', 8);
% hold on;
% plot(C(:,1), C(:,2), 'kx', 'MarkerSize',15, 'LineWidth',3);
% axis equal;
% title('Points Moved to 4 Cluster Centers');