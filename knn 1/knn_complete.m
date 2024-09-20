clear all
clc
close all

% Import the movie ratings data
data = readtable('movieknn.csv'); % Load the data as a table

% Display the variable names to check
disp(data.Properties.VariableNames);

% Get unique user IDs and create a mapping
[unique_users, ~, user_indices] = unique(data.Var1); 
num_users = length(unique_users); % Number of unique users
num_movies = height(data);         % Use the number of rows as movie count

% Initialize the user-movie matrix
user_movie_matrix = zeros(num_users, num_movies);

% Populate the user-item matrix
for i = 1:height(data)
    user_index = user_indices(i); % Map Var1 to the continuous user index
    user_movie_matrix(user_index, i) = data.rating(i); % Use mapped index
end

% Normalize features
X_norm = featureNormalize(user_movie_matrix); % Ensure featureNormalize is defined

% Split the dataset into training and testing sets
num_train = floor(0.8 * size(X_norm, 1));
Xtrain = X_norm(1:num_train, :);
Xtest = X_norm(num_train+1:end, :);

% KNN Algorithm
k = 3; % Define the number of neighbors

% Predict ratings for the test data
ypred = knn(Xtrain, Xtrain, k, Xtest); % Using KNN to predict ratings

% Display predicted ratings
disp('Predicted Ratings for the Test Set:');
disp(ypred);

% Specify the user for whom to recommend movies
user_id = 1; % Change this to the desired user ID
recommended_movies = find(ypred(user_id, :) > 3); % Recommend movies with predicted rating > 3

% Display recommended movies along with names
disp('Recommended Movies for User:');

% Iterate through recommended movies and display their names
for i = 1:length(recommended_movies)
    movie_index = recommended_movies(i);
    disp(data.Movie_name(movie_index)); % Display movie names
end
