function [CON, A, P, R, F1] = confusionmatrix(ypred, yactual)
%CONFUSIONMATRIX Computes the confusion matrix and associated metrics.
%   [CON, A, P, R, F1] = confusionmatrix(ypred, yactual) computes the 
%   confusion matrix (CON), accuracy (A), precision (P), recall (R),
%   and F1-score (F1) based on the predicted (ypred) and actual (yactual) labels.

% Initialize variables to hold counts for TP, TN, FP, FN
TP = 0; % True Positive
TN = 0; % True Negative
FP = 0; % False Positive
FN = 0; % False Negative

% Calculate TP, TN, FP, FN based on predictions and actual values
for i = 1:length(ypred)
    if yactual(i) == 1 && ypred(i) == 1
        TP = TP + 1; % True Positive (TP)
    elseif yactual(i) == 0 && ypred(i) == 0
        TN = TN + 1; % True Negative (TN)
    elseif yactual(i) == 0 && ypred(i) == 1
        FP = FP + 1; % False Positive (FP)
    elseif yactual(i) == 1 && ypred(i) == 0
        FN = FN + 1; % False Negative (FN)
    end
end

% Confusion matrix
CON = [TP, FP; FN, TN]; % Rows: actual values, Columns: predicted values

% Total number of samples
total = TP + TN + FP + FN;

% Accuracy
if total > 0
    A = (TP + TN) / total * 100; % Percentage predicted correctly
else
    A = 0; % Avoid division by zero
end

% Precision (P) = True Positives / (True Positives + False Positives)
if (TP + FP) > 0
    P = TP / (TP + FP);
else
    P = 0; % Avoid division by zero
end

% Recall (R) = True Positives / (True Positives + False Negatives)
if (TP + FN) > 0
    R = TP / (TP + FN);
else
    R = 0; % Avoid division by zero
end

% F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
if (P + R) > 0
    F1 = 2 * P * R / (P + R);
else
    F1 = 0; % Avoid division by zero
end

end
