#include "knn.h"
#include <algorithm>
#include <cmath>

struct Neighbor {
    float distance;
    int label;

    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

int predict_knn(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, const std::vector<float>& query, int k) {
    std::vector<Neighbor> distances;

    k = std::min((int)features.size(), k);
    for (size_t i = 0; i < features.size(); ++i) {
        float dist = 0.0f;
        for (size_t j = 0; j < query.size(); ++j) {
            float diff = features[i][j] - query[j];
            dist += diff * diff;
        }
        distances.push_back({ std::sqrt(dist), labels[i] });
    }

    std::nth_element(distances.begin(), distances.begin() + k, distances.end());

    int count0 = 0, count1 = 0;
    for (int i = 0; i < k; ++i) {
        if (distances[i].label == 0) count0++;
        else count1++;
    }

    return (count1 > count0) ? 1 : 0;
}
