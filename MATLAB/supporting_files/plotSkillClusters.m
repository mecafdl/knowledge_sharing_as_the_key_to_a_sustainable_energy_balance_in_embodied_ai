function plotSkillClusters(theCollective)
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
% %% MATLAB Script to Plot N 2D Points Separated into M Clusters
% 
% % Define the number of points (n) and the number of clusters (m)
% n = 500; % Total number of 2D points
% m = 5;   % Number of desired clusters
% 
%     n           = theCollective.TotalSkills;          % total points
%     m           = theCollective.TotalSkillClusters;            % clusters
% 
% 
% % --- 1. Generate M Cluster Centers ---
% % We'll randomly distribute the cluster centers within a certain range.
% % For better visualization, let's keep them somewhat separated.
% rng('default'); % For reproducibility of random numbers
% clusterCenters = 10 * rand(m, 2); % m centers, each with x and y coordinates
% 
% % --- 2. Generate N 2D Points and Assign to Clusters ---
% % Each point will be generated around one of the cluster centers with some noise.
% points = zeros(n, 2);   % Preallocate matrix for points (x, y)
% clusterIDs = zeros(n, 1); % Preallocate vector for cluster assignments
% 
% pointsPerCluster = floor(n / m); % Roughly distribute points evenly
% currentPointIdx = 1;
% 
% for i = 1:m
%     % Number of points for the current cluster
%     if i == m
%         numPointsInCurrentCluster = n - (m - 1) * pointsPerCluster;
%     else
%         numPointsInCurrentCluster = pointsPerCluster;
%     end
% 
%     % Generate points around the current cluster center
%     % Add some random noise (e.g., standard deviation of 1.5)
%     noise = 1.5 * randn(numPointsInCurrentCluster, 2);
% 
%     % Assign points
%     points(currentPointIdx : currentPointIdx + numPointsInCurrentCluster - 1, :) = ...
%         repmat(clusterCenters(i, :), numPointsInCurrentCluster, 1) + noise;
% 
%     % Assign cluster ID
%     clusterIDs(currentPointIdx : currentPointIdx + numPointsInCurrentCluster - 1) = i;
% 
%     currentPointIdx = currentPointIdx + numPointsInCurrentCluster;
% end
% 
% % --- 3. Plot the Points, Colored by Cluster ---
% % Use gscatter for convenient plotting of grouped data.
% % It automatically assigns different colors to different groups.
% 
% figure; % Create a new figure window
% 
% % gscatter(X, Y, G) creates a scatter plot of X and Y, grouping the data
% % by the grouping variable G.
% % 'filled' makes the markers filled.
% % 'LineWidth', 1.5 sets the edge line width for the markers.
% h = gscatter(points(:,1), points(:,2), clusterIDs, [], 'o', 8, 'on', ...
%              'X-coordinate', 'Y-coordinate');
% 
% % Customize plot appearance
% title(sprintf('Scatter Plot of %d Points in %d Clusters', n, m));
% xlabel('X-coordinate');
% ylabel('Y-coordinate');
% grid on;
% axis equal; % Ensures that the aspect ratio is equal for x and y axes
% 
% % You can further customize properties of the scatter series objects if needed.
% % For example, to change marker edge colors for each cluster, you would do:
% % For the 'MarkerEdgeColor', gscatter by default makes the edge color the same
% % as the face color. To explicitly change them for each series:
% for i = 1:m
%     % Get the scatter series object for the current cluster
%     % h(i) refers to the scatter object for the i-th group
% 
%     % Example: Set a specific edge color for each cluster
%     % You'll need to define your desired edge colors if you don't want them
%     % to match the face colors automatically assigned by gscatter.
%     % For instance, let's make all edges black:
%     h(i).MarkerEdgeColor = [0 0 0]; % Black
%     h(i).LineWidth = 1; % Adjust line width for visibility
% end
% 
% % Optional: Display cluster centers (larger markers)
% hold on; % Keep the current plot
% scatter(clusterCenters(:,1), clusterCenters(:,2), 200, 'k', 'x', 'LineWidth', 2);
% text(clusterCenters(:,1) + 0.5, clusterCenters(:,2), string(1:m)', 'Color', 'k', 'FontSize', 10);
% hold off;
% 
% legend('Location', 'bestoutside'); % Adjust legend position

% %% MATLAB Script to Plot N 2D Points Separated into M Clusters
% 
% % Define the number of clusters (m) and the number of points per cluster
% m = 4;   % Number of desired clusters
% pointsPerCluster = 128; % Number of 2D points in EACH cluster
% 
% % Calculate the total number of points based on the inputs
% n = m * pointsPerCluster; % Total number of 2D points
% 
% % --- 1. Generate M Cluster Centers ---
% % We'll randomly distribute the cluster centers within a certain range.
% % For better visualization, let's keep them somewhat separated.
% rng('default'); % For reproducibility of random numbers
% clusterCenters = 10 * rand(m, 2); % m centers, each with x and y coordinates
% 
% % --- 2. Generate N 2D Points and Assign to Clusters ---
% % Each point will be generated around one of the cluster centers with some noise.
% points = zeros(n, 2);   % Preallocate matrix for points (x, y)
% clusterIDs = zeros(n, 1); % Preallocate vector for cluster assignments
% 
% currentPointIdx = 1;
% 
% for i = 1:m
%     % The number of points for the current cluster is now a direct input
%     numPointsInCurrentCluster = pointsPerCluster;
% 
%     % Generate points around the current cluster center
%     % Add some random noise (e.g., standard deviation of 1.5)
%     noise = 1.5 * randn(numPointsInCurrentCluster, 2);
% 
%     % Assign points
%     points(currentPointIdx : currentPointIdx + numPointsInCurrentCluster - 1, :) = ...
%         repmat(clusterCenters(i, :), numPointsInCurrentCluster, 1) + noise;
% 
%     % Assign cluster ID
%     clusterIDs(currentPointIdx : currentPointIdx + numPointsInCurrentCluster - 1) = i;
% 
%     currentPointIdx = currentPointIdx + numPointsInCurrentCluster;
% end
% 
% % --- 3. Plot the Points, Colored by Cluster ---
% % Use gscatter for convenient plotting of grouped data.
% % It automatically assigns different colors to different groups.
% 
% figure; % Create a new figure window
% 
% % gscatter(X, Y, G) creates a scatter plot of X and Y, grouping the data
% % by the grouping variable G.
% % 'filled' makes the markers filled.
% % 'LineWidth', 1.5 sets the edge line width for the markers.
% h = gscatter(points(:,1), points(:,2), clusterIDs, [], 'o', 8, 'on', ...
%              'X-coordinate', 'Y-coordinate');
% 
% % Customize plot appearance
% title(sprintf('Scatter Plot of %d Points in %d Clusters (%d points/cluster)', n, m, pointsPerCluster));
% xlabel('X-coordinate');
% ylabel('Y-coordinate');
% grid on;
% axis equal; % Ensures that the aspect ratio is equal for x and y axes
% 
% % You can further customize properties of the scatter series objects if needed.
% % For example, to change marker edge colors for each cluster, you would do:
% % For the 'MarkerEdgeColor', gscatter by default makes the edge color the same
% % as the face color. To explicitly change them for each series:
% for i = 1:m
%     % Get the scatter series object for the current cluster
%     % h(i) refers to the scatter object for the i-th group
% 
%     % Example: Set a specific edge color for each cluster
%     % You'll need to define your desired edge colors if you don't want them
%     % to match the face colors automatically assigned by gscatter.
%     % For instance, let's make all edges black:
%     h(i).MarkerEdgeColor = [0 0 0]; % Black
%     h(i).LineWidth = 1; % Adjust line width for visibility
% end
% 
% % Optional: Display cluster centers (larger markers)
% hold on; % Keep the current plot
% scatter(clusterCenters(:,1), clusterCenters(:,2), 200, 'k', 'x', 'LineWidth', 2);
% text(clusterCenters(:,1) + 0.5, clusterCenters(:,2), string(1:m)', 'Color', 'k', 'FontSize', 10);
% hold off;
% 
% legend('Location', 'bestoutside'); % Adjust legend position


%% MATLAB Script to Plot N 2D Points Separated into M Clusters

% Define the number of points per cluster
pointsPerCluster = theCollective.SkillsPerCluster; % Number of 2D points in EACH cluster

% --- 1. Define the Similarity Matrix and Derive Cluster Centers ---
% For demonstration, let's create a synthetic similarity matrix.
% In a real application, this matrix would be an input based on your data.
% A similarity matrix S(i,j) indicates how similar cluster i is to cluster j.
% Higher values mean more similarity.
% We'll assume a symmetric matrix with 1s on the diagonal (perfect self-similarity).

% Let's define a 5x5 similarity matrix for 5 clusters (m=5)
% You would replace this with your actual similarity matrix.
% Example:
% S = [ 1.0  0.1  0.2  0.7  0.0;  % Cluster 1
%       0.1  1.0  0.8  0.1  0.2;  % Cluster 2
%       0.2  0.8  1.0  0.1  0.6;  % Cluster 3
%       0.7  0.1  0.1  1.0  0.0;  % Cluster 4
%       0.0  0.2  0.6  0.0  1.0 ]; % Cluster 5

% Let's create a more structured example for better visualization of derived centers:
% S = [ 1.0  0.2  0.1  0.8  0.1;
%       0.2  1.0  0.7  0.1  0.6;
%       0.1  0.7  1.0  0.1  0.9;
%       0.8  0.1  0.1  1.0  0.0;
%       0.1  0.6  0.9  0.0  1.0 ];

S = theCollective.ClusterSimilarityMatrix + eye(theCollective.TotalSkillClusters);

% Ensure the similarity matrix is square
if size(S, 1) ~= size(S, 2)
    error('The similarity matrix must be square.');
end

% The number of clusters (m) is now determined by the size of the similarity matrix
m = theCollective.TotalSkillClusters;

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
n = m * pointsPerCluster; % Total number of 2D points

% --- 2. Generate N 2D Points and Assign to Clusters ---
% Each point will be generated around one of the cluster centers with some noise.
points = zeros(n, 2);   % Preallocate matrix for points (x, y)
clusterIDs = zeros(n, 1); % Preallocate vector for cluster assignments

currentPointIdx = 1;

for i = 1:m
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
if m > size(clusterColors, 1)
    % Generate more distinct colors if 'm' exceeds the predefined list
    % This requires the 'distinguishable_colors' function from MATLAB File Exchange,
    % or you can use a built-in colormap like `clusterColors = hsv(m);`
    % For this example, we'll use hsv(m) for simplicity if distinguishable_colors isn't available.
    clusterColors = hsv(m); 
end



% % --- Plotting the shaded areas (convex hulls) first ---
% % This ensures the scatter points and centers are visible on top
% for i = 1:m
%     % Get points belonging to the current cluster
%     clusterPoints = points(clusterIDs == i, :);
% 
%     % Check if there are enough points to form a convex hull
%     if size(clusterPoints, 1) >= 3
%         % Find the convex hull of the cluster points
%         k = convhull(clusterPoints(:,1), clusterPoints(:,2));
% 
%         % Fill the convex hull with the cluster color, with transparency
%         fill(clusterPoints(k,1), clusterPoints(k,2), clusterColors(i,:), ...
%              'FaceAlpha', 0.2, 'EdgeColor', clusterColors(i,:), 'LineWidth', 0.5);
%     end
%     hold on
% end

% --- Plotting the shaded areas (alpha shapes) first for smooth borders ---
% This ensures the scatter points and centers are visible on top
% 'alpha' parameter controls the "tightness" of the boundary.
% A smaller alpha value creates a tighter boundary; a larger value creates a looser one.
% Adjust this value based on your data's spread.
alpha_value = 5.0; 

for i = 1:m
    % Get points belonging to the current cluster
    clusterPoints = points(clusterIDs == i, :);
    
    % Check if there are enough points to form an alpha shape
    if size(clusterPoints, 1) >= 3
        % Create an alphaShape object
        shp = alphaShape(clusterPoints(:,1), clusterPoints(:,2), alpha_value);
        
        % Plot the alpha shape
        plot(shp, 'FaceColor', clusterColors(i,:), ...
                 'FaceAlpha', 0.2, ...
                 'EdgeColor', 'none', ...
                 'LineWidth', 0.5);
    end
    hold on
end


hold on
% gscatter(X, Y, G, colors, markers, sizes, 'on', xlabel, ylabel)
% The 'colors' argument is now used to specify the colors for each group.
h = gscatter(points(:,1), points(:,2), clusterIDs, clusterColors(1:m,:), 'o', 5, 'filled');

% Customize plot appearance
title(sprintf('Scatter Plot of %d Points in %d Clusters (%d points/cluster) from Similarity Matrix', n, m, pointsPerCluster));
xlabel('Dimension 1 (from MDS)');
ylabel('Dimension 2 (from MDS)');
grid on;
axis off; % Ensures that the aspect ratio is equal for x and y axes

% Customize marker edge colors for each cluster
% Here, we explicitly set all marker edges to black for better contrast.
for i = 1:m
    % h(i) refers to the scatter object for the i-th group
    h(i).MarkerEdgeColor = [0 0 0]; % Black edge
    h(i).LineWidth = 1; % Adjust line width for visibility
end

% Optional: Display cluster centers (larger markers)
hold on; % Keep the current plot
% Adjust center marker size and style for derived centers
scatter(clusterCenters(:,1), clusterCenters(:,2), 250, 'k', 'pentagram', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 2);
text(clusterCenters(:,1) + 0.1, clusterCenters(:,2) + 0.1, string(1:m)', 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
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