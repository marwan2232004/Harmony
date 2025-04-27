#include "../core/stacking/stacking_classifier.hpp"
#include "../core/stacking/estimators.hpp"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include "../tools/tqdm.cpp"

// Logging constants
const std::string COLOR_GREEN = "\033[32m";
const std::string COLOR_CYAN = "\033[36m";
const std::string COLOR_YELLOW = "\033[33m";
const std::string COLOR_RESET = "\033[0m";

namespace fs = std::filesystem;

void printColored(const std::string& message, const std::string& color) {
    std::cout << color << message << COLOR_RESET << std::endl;
}

struct Dataset {
    Eigen::MatrixXd X;
    Eigen::VectorXi y;
};

Dataset loadTSV(const std::string& path) {
    Dataset dataset;
    std::ifstream file(path);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    // First pass to count rows and columns
    int rows = 0;
    int cols = 0;
    while (std::getline(file, line)) rows++;
    file.clear();
    file.seekg(0);
    std::getline(file, line); // Skip header again
    std::istringstream iss(line);
    std::string cell;
    while (std::getline(iss, cell, '\t')) cols++;
    cols -= 2; // Last 2 columns are label
    
    // Initialize matrices
    dataset.X.resize(rows, cols);
    dataset.y.resize(rows);
    
    // Second pass to load data
    file.seekg(0);
    std::getline(file, line); // Skip header
    int row_idx = 0;
    Tqdm tqdm(rows, "📂 Loading " + fs::path(path).filename().string());
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int col = 0; col < cols; ++col) {
            std::string val;
            std::getline(iss, val, '\t');
            dataset.X(row_idx, col) = std::stod(val);
        }
        std::string ageLabel;
        std::getline(iss, ageLabel, '\t');
        std::string genderLabel;
        std::getline(iss, genderLabel, '\t');
        // Label is either TWENTIES or FIFTIES
        // Convert to 0 or 1
        if (ageLabel == "twenties") {
            dataset.y(row_idx) = genderLabel == "female" ? 0 : 1;
        } else if (ageLabel == "fifties") {
            dataset.y(row_idx) = genderLabel == "female" ? 2 : 3;
        } else {
            std::cerr << "Unknown age label: " << ageLabel << std::endl;
            continue;
        }
        
        tqdm.update();
        row_idx++;
    }
    tqdm.finish();
    return dataset;
}

double calculateAccuracy(const Eigen::VectorXi& y_true, const Eigen::VectorXi& y_pred) {
    int correct = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == y_pred(i)) correct++;
    }
    return static_cast<double>(correct) / y_true.size() * 100.0;
}

int main() {
    // Configuration
    const std::string train_path = "data/features/train.tsv";
    const std::string test_path = "data/features/test.tsv";
    const int n_folds = 5;
    const unsigned seed = 42;

    // Pretty header
    std::cout << "\n🎯 " << COLOR_CYAN << "Starting Stacking Classifier Training" << COLOR_RESET << " 🎯\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "⚙️  " << COLOR_YELLOW << "Model Configuration:\n" << COLOR_RESET;
    std::cout << "▸ Base Models:\n";
    std::cout << "   - SVM with RBF Kernel (C=1000, gamma=0.0001)\n";
    std::cout << "   - Extra Trees (200 trees, min_leaf=5)\n";
    std::cout << "   - Random Forest (300 trees, min_leaf=5)\n";
    std::cout << "   - K-Nearest Neighbors (k=3)\n";
    std::cout << "▸ Meta Model: Logistic Regression\n";
    std::cout << "▸ Cross-Validation Folds: " << n_folds << "\n";
    std::cout << "▸ Random Seed: " << seed << "\n";
    std::cout << std::string(60, '-') << "\n\n";

    // Load data
    printColored("🚀 Loading training data...", COLOR_GREEN);
    auto train_start = std::chrono::high_resolution_clock::now();
    Dataset train_data = loadTSV(train_path);
    
    printColored("\n🚀 Loading test data...", COLOR_GREEN);
    Dataset test_data = loadTSV(test_path);
    auto load_end = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n📊 Dataset Statistics:\n";
    std::cout << "▸ Training Samples: " << COLOR_CYAN << train_data.X.rows() << COLOR_RESET << "\n";
    std::cout << "▸ Test Samples:     " << COLOR_CYAN << test_data.X.rows() << COLOR_RESET << "\n";
    std::cout << "▸ Features:         " << COLOR_CYAN << train_data.X.cols() << COLOR_RESET << "\n";
    std::cout << std::string(60, '-') << "\n\n";

    // Initialize models
    printColored("⚡ Initializing models...", COLOR_GREEN);
    std::vector<std::unique_ptr<BaseEstimator>> base_models;
    base_models.push_back(std::make_unique<harmony::SVM>(1000, 0.0001));
    // base_models.push_back(std::make_unique<harmony::ExtraTrees>(200, 5));
    base_models.push_back(std::make_unique<harmony::RandomForest>(300, 5));
    base_models.push_back(std::make_unique<harmony::KNN>(3));
    auto meta_model = std::make_unique<harmony::LR>(0.01);

    // Create stacker
    StackingClassifier stacker(
        std::move(base_models),
        std::move(meta_model),
        n_folds,
        seed
    );

    // Training
    printColored("🏋️  Training stacking classifier...", COLOR_GREEN);
    auto train_start_time = std::chrono::high_resolution_clock::now();
    stacker.fit(train_data.X, train_data.y);
    auto train_end_time = std::chrono::high_resolution_clock::now();

    // Prediction
    printColored("\n🔮 Making predictions...", COLOR_GREEN);
    Eigen::VectorXi y_pred;
    Tqdm predict_tqdm(test_data.X.rows(), "Predicting");
    stacker.predict(test_data.X, y_pred);
    predict_tqdm.finish();

    // Calculate accuracy
    double accuracy = calculateAccuracy(test_data.y, y_pred);
    auto total_end = std::chrono::high_resolution_clock::now();

    // Final report
    auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(train_end_time - train_start_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - train_start);
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    printColored("🎉 Stacking Classifier Results", COLOR_CYAN);
    std::cout << "⏱️  Training Time:    " << train_duration.count() << " seconds\n";
    std::cout << "⏱️  Total Time:       " << total_duration.count() << " seconds\n";
    std::cout << "📊 Test Accuracy:    " << COLOR_GREEN 
              << std::fixed << std::setprecision(2) << accuracy << "%" << COLOR_RESET << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    return 0;
}