#include "../core/stacking/stacking_classifier.hpp"
#include "../core/stacking/estimators.hpp"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <map>
#include <ctime>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/core/data/load.hpp>
#include "../tools/tqdm.cpp"
#include "../utils/logger.hpp"
#include "../utils/arg_parser.hpp"

// Logging constants
const std::string COLOR_GREEN = "\033[32m";
const std::string COLOR_CYAN = "\033[36m";
const std::string COLOR_YELLOW = "\033[33m";
const std::string COLOR_RESET = "\033[0m";
const std::string COLOR_RED = "\033[31m";

namespace fs = std::filesystem;
using TYPE = harmony::ArgParser::TYPE;
using COLOR = harmony::Logger::COLOR;

harmony::Logger &logger = harmony::Logger::getInstance();

std::string get(const std::string& arg, const std::string& paramName) {
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
    
    // First pass to count rows and columns
    int rows = 0;
    int cols = 0;
    while (std::getline(file, line)) rows++;
    file.clear();
    file.seekg(0);
    std::getline(file, line);
    std::istringstream iss(line);
    std::string cell;
    while (std::getline(iss, cell, '\t')) cols++;
    logger.log("🔄 Loaded " + std::to_string(rows) + " rows and " + std::to_string(cols) + " columns from " + path, COLOR::GREEN);
    // Print number of columns
    logger.log("🔄 Number of columns: " + std::to_string(cols), COLOR::GREEN);
    cols -= 2; // Last 2 columns are label
    
    // Initialize matrices
    dataset.X.resize(rows, cols);
    dataset.y.resize(rows);
    
    // Second pass to load data
    file.seekg(0);
    int row_idx = 0;
    harmony::Logger::ProgressBar progressBar(rows, "📂 Loading " + fs::path(path).filename().string());
    
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
            dataset.y(row_idx) = (genderLabel == "male") ? 0 : 1;
        } else if (target == "age") {
            dataset.y(row_idx) = (ageLabel == "twenties") ? 0 : 1;
        } else {
            // Combine predictions to get final class (0-3)
            // 0: Male + Twenties
            // 1: Female + Twenties
            // 2: Male + Fifties
            // 3: Female + Fifties
            // if (ageLabel == "twenties") {
            //     dataset.y(row_idx) = genderLabel == "male" ? 0 : 1;
            // } else if (ageLabel == "fifties") {
            //     dataset.y(row_idx) = genderLabel == "male" ? 2 : 3;
            // }
            int classLabel = 0;
            if (ageLabel == "twenties" && genderLabel == "male") {
                classLabel = 0;
            } else if (ageLabel == "twenties" && genderLabel == "female") {
                classLabel = 1;
            } else if (ageLabel == "fifties" && genderLabel == "male") {
                classLabel = 2;
            } else if (ageLabel == "fifties" && genderLabel == "female") {
                classLabel = 3;
            }
            int ageCode = (ageLabel == "twenties") ? 0 : 1;
            int genderCode = (genderLabel == "male") ? 0 : 1;
            int label = ageCode * 2 + genderCode;
            assert(label == classLabel);
            dataset.y(row_idx) = label;
        }
        
        progressBar.update();
        row_idx++;
    }
    progressBar.finish();
    return dataset;
}

double calculateAccuracy(const Eigen::VectorXi& y_true, const Eigen::VectorXi& y_pred) {
    int correct = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == y_pred(i)) correct++;
    }
    logger.log("✅ Accuracy: " + std::to_string(correct) + "/" + std::to_string(y_true.size()), COLOR::GREEN);
    return static_cast<double>(correct) / y_true.size() * 100.0;
}

// Function to ensure a directory exists
void ensureDirectoryExists(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}


int main(int argc, char* argv[]) {
    omp_set_num_threads(std::min(12, omp_get_num_procs()));
    // Configuration
    std::string train_path = "data/features/train.tsv";
    std::string test_path = "data/features/test.tsv";
    std::string target = "both";
    int svm_c = 1000;
    double svm_gamma = 0.0001;
    int rf_trees = 700;
    int knn_k = 5;
    std::string knn_metric = "euclidean";
    int n_folds = 5;
    unsigned seed = 42;
    int nn_hidden1 = 64;
    int nn_hidden2 = 32;

    harmony::ArgParser parser(argc, argv);
    parser.addOption("train-path", "Path to training data", train_path);
    parser.addOption("test-path", "Path to test data", test_path);
    parser.addOption("target", "Prediction target: 'gender', 'age', or 'both'", target);
    parser.addOption("svm-c", "SVM C parameter", svm_c);
    parser.addOption("svm-gamma", "SVM gamma parameter", svm_gamma);
    parser.addOption("rf-trees", "Random Forest number of trees", rf_trees);
    parser.addOption("knn-k", "KNN number of neighbors", knn_k);
    parser.addOption("knn-metric", "KNN distance metric (euclidean or manhattan)", knn_metric);
    parser.addOption("nn-hidden1", "Neural Network first hidden layer units", nn_hidden1);
    parser.addOption("nn-hidden2", "Neural Network second hidden layer units", nn_hidden2);
    parser.addOption("n-folds", "Cross-validation folds", n_folds);
    parser.addOption("seed", "Random seed", seed);

    // Parse command line arguments
    parser.parse();
    train_path = parser.get<std::string>("train-path");
    test_path = parser.get<std::string>("test-path");
    target = parser.get<std::string>("target");
    svm_c = parser.get<int>("svm-c");
    svm_gamma = parser.get<double>("svm-gamma");
    rf_trees = parser.get<int>("rf-trees");
    knn_k = parser.get<int>("knn-k");
    knn_metric = parser.get<std::string>("knn-metric");
    nn_hidden1 = parser.get<int>("nn-hidden1");
    nn_hidden2 = parser.get<int>("nn-hidden2");
    n_folds = parser.get<int>("n-folds");
    seed = parser.get<unsigned>("seed");

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
    std::cout << "   - Extra Trees (400 trees, min_leaf=5)\n";
    std::cout << "   - Neural Network (" << nn_hidden1 << ", " << nn_hidden2 << " hidden units)\n";
    std::cout << "▸ Meta Model: Logistic Regression\n";
    std::cout << "▸ Cross-Validation Folds: " << n_folds << "\n";
    std::cout << "▸ Random Seed: " << seed << "\n";
    std::cout << "▸ Prediction Target: " << target << "\n";
    std::cout << std::string(60, '-') << "\n\n";

    // Load data
    logger.log("🚀 Loading training data...", COLOR::GREEN);
    auto train_start = std::chrono::high_resolution_clock::now();
    Dataset train_data = loadTSV(train_path, target);
    
    logger.log("\n🚀 Loading test data...", COLOR::GREEN);
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
    logger.log("⚡ Initializing models...", COLOR::GREEN);
    std::vector<std::unique_ptr<BaseEstimator>> base_models;
    base_models.push_back(std::make_unique<harmony::SVM_ML>(svm_c, svm_gamma));
    // base_models.push_back(std::make_unique<harmony::ExtraTrees>(400, 5, nClasses));
    // base_models.push_back(std::make_unique<harmony::RandomForest>(rf_trees, 5, nClasses));
    base_models.push_back(std::make_unique<harmony::KNN>(knn_k, knn_metric));
    // base_models.push_back(std::make_unique<harmony::NeuralNet>(nn_hidden1, nn_hidden2, nClasses));
    auto meta_model = std::make_unique<harmony::LR>(0.001, nClasses);

    // Create stacker
    StackingClassifier stacker(
        std::move(base_models),
        std::move(meta_model),
        n_folds,
        seed
    );

    // Training
    logger.log("🏋️  Training stacking classifier...", COLOR::GREEN);
    auto train_start_time = std::chrono::high_resolution_clock::now();
    stacker.fit(train_data.X, train_data.y);
    auto train_end_time = std::chrono::high_resolution_clock::now();

    // Prediction
    logger.log("\n🔮 Making predictions...", COLOR::GREEN);
    Eigen::VectorXi y_pred;
    harmony::Logger::ProgressBar progressBar(test_data.X.rows(), "Predicting");
    stacker.predict(test_data.X, y_pred);
    progressBar.finish();

    // Calculate accuracy
    double accuracy = calculateAccuracy(test_data.y, y_pred);
    auto total_end = std::chrono::high_resolution_clock::now();

    // Final report
    auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(train_end_time - train_start_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - train_start);

    
    std::cout << "\n" << std::string(60, '=') << "\n";
    logger.log("🎉 Stacking Classifier Results", COLOR::CYAN);
    std::cout << "⏱️  Training Time:    " << train_duration.count() << " seconds\n";
    std::cout << "⏱️  Total Time:       " << total_duration.count() << " seconds\n";
    std::cout << "📊 Test Accuracy:    " << COLOR_GREEN 
              << std::fixed << std::setprecision(2) << accuracy << "%" << COLOR_RESET << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Save the trained models
    std::string modelDir = "models/" + target + "_" + std::to_string(std::time(nullptr));
    ensureDirectoryExists(modelDir);

    logger.log("💾 Saving models to " + modelDir + "...", COLOR::GREEN);
    if (stacker.saveModels(modelDir)) {
        logger.log("✅ Models saved successfully!", COLOR::GREEN);
        
        // Save a summary file with model parameters and accuracy
        std::ofstream summary(modelDir + "/summary.txt");
        if (summary.is_open()) {
            summary << "# Stacking Model Summary\n\n";
            summary << "Target: " << target << "\n";
            summary << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n\n";
            summary << "## Parameters\n";
            summary << "SVM C: " << svm_c << "\n";
            summary << "SVM gamma: " << std::fixed << std::setprecision(6) << svm_gamma << "\n";
            summary << "Random Forest trees: " << rf_trees << "\n";
            summary << "KNN k: " << knn_k << "\n";
            summary << "KNN metric: " << knn_metric << "\n";
            summary << "Neural Network hidden1: " << nn_hidden1 << "\n";
            summary << "Neural Network hidden2: " << nn_hidden2 << "\n";
            summary << "Cross-validation folds: " << n_folds << "\n";
            summary.close();
        }
    } else {
        logger.log("❌ Failed to save models!", COLOR::RED);
    }
    return 0;
}