function y = knn(Xtrain, ytrain, k, Xtest)
% KNN function to predict labels for Xtest based on Xtrain and ytrain
% Inputs:
%   Xtrain: Training data (matrix of features)
%   ytrain: Training labels (vector)
%   k: Number of nearest neighbors
%   Xtest: Test data (matrix of features for which predictions are required)
% Outputs:
%   y: Predicted labels for Xtest

% Initialize output vector
y = zeros(size(Xtest, 1), 1);

% Loop over each test data point
for i = 1:size(Xtest, 1)
    nx = Xtest(i, :);  % Current test data point

    % Calculate Euclidean distance from the test point to all training points
    distances = sqrt(sum((Xtrain - nx).^2, 2));  % Vectorized distance calculation

    % Get the indices of the k nearest neighbors
    [~, sorted_indices] = sort(distances);  % Sort distances and get indices
    nearest_labels = ytrain(sorted_indices(1:k));  % Get labels of the k nearest neighbors

    % Count the number of occurrences of each class (0 and 1)
    count_ones = sum(nearest_labels == 1);
    count_zeros = sum(nearest_labels == 0);
    
    % Predict the class based on the majority vote
    if count_ones >= count_zeros
        y(i) = 1;  % Predict 1 if there are more or equal ones
    else
        y(i) = 0;  % Predict 0 otherwise
    end
end

end

