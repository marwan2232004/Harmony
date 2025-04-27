#include "knn.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

struct Neighbor {
    float distance;
    int label;
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Feature size mismatch");
    }
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float manhattan_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Feature size mismatch");
    }
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

int predict_knn(const std::vector<std::vector<float>>& features,
                const std::vector<int>& labels,
                const std::vector<float>& query,
                int k,
                const std::string& metric) {
    if (features.empty() || features.size() != labels.size()) {
        throw std::invalid_argument("Invalid training data");
    }
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    // Get distance function
    auto distance = (metric == "manhattan") ? manhattan_distance : euclidean_distance;

    // Use priority queue to maintain k smallest neighbors
    std::priority_queue<Neighbor> neighbors;

    // Calculate distances and maintain k nearest
    for (size_t i = 0; i < features.size(); ++i) {
        float dist = distance(features[i], query);
        neighbors.push({dist, labels[i]});
        
        // Remove furthest neighbor if we exceed k
        if (neighbors.size() > static_cast<size_t>(k)) {
            neighbors.pop();
        }
    }

    // Handle case where k is larger than dataset
    k = std::min(k, static_cast<int>(neighbors.size()));

    // Count votes using map to handle multiple classes
    std::unordered_map<int, int> class_counts;
    while (!neighbors.empty()) {
        class_counts[neighbors.top().label]++;
        neighbors.pop();
    }

    // Find majority class
    int max_count = 0;
    int prediction = -1;
    for (const auto& [label, count] : class_counts) {
        if (count > max_count) {
            max_count = count;
            prediction = label;
        }
    }

    return prediction;
}