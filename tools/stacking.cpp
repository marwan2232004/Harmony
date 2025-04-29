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

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --train-path=<path>      Path to training data (default: data/features/train.tsv)\n";
    std::cout << "  --test-path=<path>       Path to test data (default: data/features/test.tsv)\n";
    std::cout << "  --target=<target>        Prediction target: 'gender', 'age', or 'both' (default: both)\n";
    std::cout << "  --svm-c=<value>          SVM C parameter (default: 1000)\n";
    std::cout << "  --svm-gamma=<value>      SVM gamma parameter (default: 0.0001)\n";
    std::cout << "  --rf-trees=<num>         Random Forest number of trees (default: 300)\n";
    std::cout << "  --knn-k=<num>            KNN number of neighbors (default: 3)\n";
    std::cout << "  --knn-metric=<metric>    KNN distance metric (euclidean or manhattan, default: euclidean)\n";
    std::cout << "  --n-folds=<num>          Cross-validation folds (default: 5)\n";
    std::cout << "  --seed=<num>             Random seed (default: 42)\n";
    std::cout << "  --help                   Show this help message\n";
}

std::string getParamValue(const std::string& arg, const std::string& paramName) {
    std::string prefix = "--" + paramName + "=";
    if (arg.substr(0, prefix.size()) == prefix) {
        return arg.substr(prefix.size());
    }
    return "";
}

struct Dataset {
    Eigen::MatrixXd X;
    Eigen::VectorXi y;
};

Dataset loadTSV(const std::string& path, const std::string& target) {
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
        if (target == "gender") {
            dataset.y(row_idx) = (genderLabel == "female") ? 0 : 1;
        } else if (target == "age") {
            dataset.y(row_idx) = (ageLabel == "twenties") ? 0 : 1;
        } else {
            if (ageLabel == "twenties") {
                dataset.y(row_idx) = genderLabel == "female" ? 0 : 1;
            } else if (ageLabel == "fifties") {
                dataset.y(row_idx) = genderLabel == "female" ? 2 : 3;
            }
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

int main(int argc, char* argv[]) {
    // Configuration
    std::string train_path = "data/features/train.tsv";
    std::string test_path = "data/features/test.tsv";
    std::string target = "both";
    int svm_c = 1000;
    double svm_gamma = 0.0001;
    int rf_trees = 700;
    int knn_k = 3;
    std::string knn_metric = "euclidean";
    int n_folds = 5;
    unsigned seed = 42;

    // Parse command line arguments
    std::vector<std::string> args(argv + 1, argv + argc);
    for (const auto& arg : args) {
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        
        std::string value;
        if (!(value = getParamValue(arg, "train-path")).empty()) {
            train_path = value;
        } else if (!(value = getParamValue(arg, "test-path")).empty()) {
            test_path = value;
        } else if (!(value = getParamValue(arg, "target")).empty()) {
            target = value;
        } else if (!(value = getParamValue(arg, "svm-c")).empty()) {
            svm_c = std::stoi(value);
        } else if (!(value = getParamValue(arg, "svm-gamma")).empty()) {
            svm_gamma = std::stod(value);
        } else if (!(value = getParamValue(arg, "rf-trees")).empty()) {
            rf_trees = std::stoi(value);
        } else if (!(value = getParamValue(arg, "knn-k")).empty()) {
            knn_k = std::stoi(value);
        } else if (!(value = getParamValue(arg, "knn-metric")).empty()) {
            knn_metric = value;
        } else if (!(value = getParamValue(arg, "n-folds")).empty()) {
            n_folds = std::stoi(value);
        } else if (!(value = getParamValue(arg, "seed")).empty()) {
            seed = std::stoi(value);
        }
    }

    // Validate target
    if (target != "gender" && target != "age" && target != "both") {
        std::cerr << "Invalid target: " << target << ". Must be 'gender', 'age', or 'both'\n";
        return 1;
    }

    // Pretty header
    std::cout << "\n🎯 " << COLOR_CYAN << "Starting Stacking Classifier Training" << COLOR_RESET << " 🎯\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "⚙️  " << COLOR_YELLOW << "Model Configuration:\n" << COLOR_RESET;
    std::cout << "▸ Base Models:\n";
    std::cout << "   - SVM with RBF Kernel (C=" << svm_c << ", gamma=" << svm_gamma << ")\n";
    std::cout << "   - Random Forest (" << rf_trees << " trees, min_leaf=5)\n";
    std::cout << "   - K-Nearest Neighbors (k=" << knn_k << ", metric=" << knn_metric << ")\n";
    std::cout << "▸ Meta Model: Logistic Regression\n";
    std::cout << "▸ Cross-Validation Folds: " << n_folds << "\n";
    std::cout << "▸ Random Seed: " << seed << "\n";
    std::cout << "▸ Prediction Target: " << target << "\n";
    std::cout << std::string(60, '-') << "\n\n";

    // Load data
    printColored("🚀 Loading training data...", COLOR_GREEN);
    auto train_start = std::chrono::high_resolution_clock::now();
    Dataset train_data = loadTSV(train_path, target);
    
    printColored("\n🚀 Loading test data...", COLOR_GREEN);
    Dataset test_data = loadTSV(test_path, target);
    auto load_end = std::chrono::high_resolution_clock::now();
    
    size_t nClasses = (target == "both" ? 4 : 2);

    std::cout << "\n📊 Dataset Statistics:\n";
    std::cout << "▸ Training Samples: " << COLOR_CYAN << train_data.X.rows() << COLOR_RESET << "\n";
    std::cout << "▸ Test Samples:     " << COLOR_CYAN << test_data.X.rows() << COLOR_RESET << "\n";
    std::cout << "▸ Features:         " << COLOR_CYAN << train_data.X.cols() << COLOR_RESET << "\n";
    std::cout << "▸ Classes:          " << COLOR_CYAN << nClasses << COLOR_RESET << "\n";
    std::cout << std::string(60, '-') << "\n\n";

    // Initialize models
    printColored("⚡ Initializing models...", COLOR_GREEN);
    std::vector<std::unique_ptr<BaseEstimator>> base_models;
    base_models.push_back(std::make_unique<harmony::SVM>(svm_c, svm_gamma));
    // base_models.push_back(std::make_unique<harmony::ExtraTrees>(400, 5, nClasses));
    // base_models.push_back(std::make_unique<harmony::RandomForest>(rf_trees, 5, nClasses));
    // base_models.push_back(std::make_unique<harmony::KNN>(knn_k, knn_metric));
    auto meta_model = std::make_unique<harmony::LR>(0.01, nClasses);

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