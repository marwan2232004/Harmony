#pragma once
#include <vector>
#include <string>

/**
 * @brief Predicts the label of a query point using K-Nearest Neighbors algorithm.
 * 
 * @param features Training data features (n_samples x n_features).
 * @param labels Training data labels (n_samples).
 * @param query Query point features (n_features).
 * @param k Number of neighbors to consider.
 * @param metric Distance metric to use (e.g., "euclidean", "manhattan").
 * 
 * @return Predicted label (0 or 1).
 */
int predict_knn(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, const std::vector<float>& query, int k, const std::string& metric);
