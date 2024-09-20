function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Initialize variables
mu = mean(X);      % Compute the mean of each feature (column-wise)
sigma = std(X);    % Compute the standard deviation of each feature
X_norm = X;        % Initialize X_norm with the original X matrix

% Normalize each feature
[m, n] = size(X);  % Get the number of rows (m) and columns (n)
for j = 1:n
    if sigma(j) ~= 0  % Avoid division by zero
        X_norm(:, j) = (X(:, j) - mu(j)) / sigma(j);  % Normalize each feature
    else
        X_norm(:, j) = X(:, j) - mu(j);  % If std is 0, center the data
    end
end

end
