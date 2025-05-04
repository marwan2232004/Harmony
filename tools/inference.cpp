#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <memory>
#include <map>
#include <unordered_map>
#include <sys/stat.h>
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <eigen3/Eigen/Dense>
// Include Harmony headers
#include "../core/stacking/stacking_classifier.hpp"
#include "../core/stacking/estimators.hpp"
#include "../core/preprocessing/audio_preprocessor.hpp"
#include "feature_extractor.h"
#include "feature_utils.h"
#include "../utils/logger.hpp"
#include "../utils/arg_parser.hpp"

namespace fs = std::filesystem;
using TYPE = harmony::ArgParser::TYPE;
using COLOR = harmony::Logger::COLOR;
using LEVEL = harmony::Logger::Level;

harmony::Logger &logger = harmony::Logger::getInstance();

std::unique_ptr<StackingClassifier> loadModelFromFile(const std::string& modelDir, const std::string& configPrefix) {

    std::string modelSubdir = configPrefix.empty() ? modelDir : modelDir + "/" + configPrefix;

    // Load configuration parameters from summary file
    std::ifstream summaryFile(modelSubdir + "/summary.txt");
    if (!summaryFile.is_open()) {
        std::cerr << "Error: Failed to open summary file" << std::endl;
        return nullptr;
    }
    auto trim = [](std::string s) {
        s.erase(0, s.find_first_not_of(" \t\r\n"));
        s.erase(s.find_last_not_of(" \t\r\n") + 1);
        return s;
    };

    std::string line;
    int svm_c = 1000;
    double svm_gamma = 0.0001;
    int rf_trees = 700;
    int knn_k = 5;
    std::string knn_metric = "euclidean";
    int n_folds = 5;
    unsigned seed = 42;
    int nn_hidden1 = 64;
    int nn_hidden2 = 32;
    int n_classes = 2;  // Default for binary classification

    std::string prefix = configPrefix.empty() ? "" : configPrefix + "_";

    while (std::getline(summaryFile, line)) {
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string value = trim(line.substr(pos + 1));
        try {
            if (key == "SVM C") svm_c = std::stoi(value);
            else if (key == "SVM gamma") svm_gamma = std::stod(value);
            else if (key == "Random Forest trees") rf_trees = std::stoi(value);
            else if (key == "KNN k") knn_k = std::stoi(value);
            else if (key == "KNN metric") knn_metric = value;
            else if (key == "Neural Network hidden1") nn_hidden1 = std::stoi(value);
            else if (key == "Neural Network hidden2") nn_hidden2 = std::stoi(value);
            else if (key == "Cross-validation folds") n_folds = std::stoi(value);
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to parse '" << key << "': '" << value << "' (" << e.what() << ")\n";
        }
    }

    // Create base models matching training configuration
    std::vector<std::unique_ptr<BaseEstimator>> base_models;
    
    // Uncomment these when needed and properly implemented
    logger.log("▸ Loading SVM model with C=" + std::to_string(svm_c) + " and gamma=" + std::to_string(svm_gamma), COLOR::RESET);
    base_models.push_back(std::make_unique<harmony::SVM_ML>(svm_c, svm_gamma));
    base_models.push_back(std::make_unique<harmony::KNN>(knn_k, knn_metric));
    // base_models.push_back(std::make_unique<harmony::RandomForest>(rf_trees, 5, n_classes));
    // base_models.push_back(std::make_unique<harmony::NeuralNet>(nn_hidden1, nn_hidden2, n_classes));
    
    // Create meta model
    logger.log("▸ Loading Logistic Regression model with lambda=0.01", COLOR::RESET);
    auto meta_model = std::make_unique<harmony::LR>(0.01, n_classes);
    
    // Create stacking classifier
    auto classifier = std::make_unique<StackingClassifier>(std::move(base_models), std::move(meta_model));
    
    // Load models
    if (!classifier->loadModels(modelSubdir)) {
        std::cerr << "Error: Failed to load " << configPrefix << " models from " << modelSubdir << std::endl;
        return nullptr;
    }
    
    return classifier;
}

struct Config {
    std::string dataDir = "data/test";
    std::string modelDir = "models";
    std::string groundTruthPath = "data/datasets/filtered_data_labeled.tsv";
    std::string mode = "combined";
    std::string genderPrefix = "gender_3";
    std::string agePrefix = "age_3";
};

class Inference {
public:
    Inference(int argc, char* argv[]) : argc(argc), argv(argv) {}

    bool initialize() {
        parseArguments();
        return verifyDirectories();
    }

    int run() {
        logger.log("🚀 Starting inference...", COLOR::GREEN);
        startTimer();
        loadClassifiers();
        auto files = getTestFiles();
        if (files.empty()) {
            logger.log("No audio files found in directory: " + config.dataDir, LEVEL::ERROR);
            return 1;
        }

        auto features = extractAllFeatures(files);
        auto predictions = predict(features);
        writeOutputs(predictions, files);
        logElapsedTime();
        return 0;
    }

private:
    int argc;
    char** argv;
    Config config;
    std::unique_ptr<StackingClassifier> classifier;
    std::unique_ptr<StackingClassifier> genderClassifier;
    std::unique_ptr<StackingClassifier> ageClassifier;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    void parseArguments() {
        harmony::ArgParser parser(argc, argv);
        parser.addOption("data-dir", "Directory containing audio files", config.dataDir);
        parser.addOption("model-dir", "Directory containing model files", config.modelDir);
        parser.addOption("ground-truth", "TSV file with ground truth labels", config.groundTruthPath);
        parser.addOption("mode", "Mode: 'combined' for separate gender/age models, 'single' for one model", config.mode);
        parser.addOption("gender-prefix", "Prefix for gender model files", config.genderPrefix);
        parser.addOption("age-prefix", "Prefix for age model files", config.agePrefix);
        parser.parse();
        config.dataDir = parser.get<std::string>("data-dir");
        config.modelDir = parser.get<std::string>("model-dir");
        config.groundTruthPath = parser.get<std::string>("ground-truth");
        config.mode = parser.get<std::string>("mode");
        config.genderPrefix = parser.get<std::string>("gender-prefix");
        config.agePrefix = parser.get<std::string>("age-prefix");

    }

    bool verifyDirectories() const {
        if (!fs::exists(config.dataDir) || !fs::is_directory(config.dataDir)) {
            logger.log("Data directory not found: " + config.dataDir, LEVEL::ERROR);
            return false;
        }
        if (!fs::exists(config.modelDir) || !fs::is_directory(config.modelDir)) {
            logger.log("Model directory not found: " + config.modelDir, LEVEL::ERROR);
            return false;
        }
        return true;
    }

    void loadClassifiers() {
        logger.log("\n⚡ Loading classifiers...", COLOR::GREEN);
        if (config.mode == "combined") {
            genderClassifier = this->loadModel(config.modelDir, config.genderPrefix);
            ageClassifier = this->loadModel(config.modelDir, config.agePrefix);
        } else if (config.mode == "single") {
            classifier = this->loadModel(config.modelDir, "");
        } else {
            logger.log("Invalid mode specified: " + config.mode, LEVEL::ERROR);
        }
    }

    std::vector<std::string> getTestFiles() const {
        std::vector<std::string> files;
        logger.log("\n📂 Searching for audio files in " + config.dataDir, COLOR::GREEN);
        for (const auto& entry : fs::directory_iterator(config.dataDir)) {
            if (!entry.is_regular_file()) continue;
            struct stat st;
            if (stat(entry.path().c_str(), &st) == 0 && st.st_size == 0) continue; // skip empty
            auto ext = entry.path().extension().string();
            if (ext == ".mp3" || ext == ".wav")
                files.push_back(entry.path().filename().string());
        }
        std::sort(files.begin(), files.end(), numericalSort);
        return files;
    }

    static bool numericalSort(const std::string& a, const std::string& b) {
        // same as original implementation
        std::string na, nb;
        size_t ia = 0, ib = 0;
        while (ia < a.size() && isdigit(a[ia])) na += a[ia++];
        while (ib < b.size() && isdigit(b[ib])) nb += b[ib++];
        if (!na.empty() && !nb.empty()) return std::stoi(na) < std::stoi(nb);
        return a < b;
    }

    std::vector<std::vector<float>> extractAllFeatures(const std::vector<std::string>& files) const {
        using namespace essentia;
        using namespace essentia::standard;
        AudioPreprocessor processor(1);
        processor.enableTrimming(false);
        processor.enableNoiseReduction(false);
        std::vector<std::vector<float>> allFeatures;
        harmony::Logger::ProgressBar progressBar(files.size(), "🔄 Extracting features", COLOR::BLUE);
        for (const auto& file : files) {
            std::string path = config.dataDir + "/" + file;
            float duration;
            std::vector<essentia::Real> buffer;
            bool ok = processor.processFile(path, "", duration, AlgorithmFactory::instance(), buffer, false);
            if (!ok || buffer.empty()) {
                allFeatures.emplace_back();
                progressBar.update();
                continue;
            }
            try {
                allFeatures.push_back(getFeatureVector("", buffer));
            } catch (...) {
                allFeatures.emplace_back();
            }
            progressBar.update();
        }
        progressBar.finish();
        return allFeatures;
    }

    std::vector<int> predict(const std::vector<std::vector<float>>& features) {
        logger.log("\n🔮 Making predictions...", COLOR::GREEN);
        int M = features.size();
        Eigen::MatrixXd X(M, features[0].size());
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < (int)features[i].size(); ++j)
                X(i, j) = features[i][j];

        std::vector<int> finalClasses(M);
        if (config.mode == "combined") {
            Eigen::VectorXi gPred(M), aPred(M);
            genderClassifier->predict(X, gPred);
            ageClassifier->predict(X, aPred);
            for (int i = 0; i < M; ++i)
                finalClasses[i] = aPred(i)*2 + gPred(i);
        } else {
            Eigen::VectorXi pred(M);
            classifier->predict(X, pred);
            for (int i = 0; i < M; ++i)
                finalClasses[i] = pred(i);
        }
        return finalClasses;
    }

    void writeOutputs(const std::vector<int>& preds, const std::vector<std::string>& files) {
        std::ofstream resF("results.txt");
        for (int cls : preds) resF << cls << "\n";
        auto duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime);

        if (fs::exists(config.groundTruthPath)) {
            auto truth = parseGroundTruth(config.groundTruthPath);
            std::ofstream accF("accuracy.txt");
            int correct = 0;
            for (size_t i = 0; i < files.size(); ++i) {
                auto it = truth.find(files[i]);
                if (it != truth.end() && it->second == preds[i]) ++correct;
                accF << files[i] << "\tTrue:" << (it!=truth.end()?it->second: -1)
                     << "\tPred:" << preds[i]
                     << "\t" << ((it!=truth.end() && it->second==preds[i])?"Correct":"Wrong")
                     << "\n";
            }
            logger.log("Accuracy: " + std::to_string(100.0*correct/files.size()) + "%", COLOR::GREEN);
        }
    }

    void startTimer() { startTime = std::chrono::high_resolution_clock::now(); }

    void logElapsedTime() const {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "Inference completed in " << dur.count() << " ms\n";
    }

    std::unique_ptr<StackingClassifier> loadModel(const std::string& dir, const std::string& prefix) const {
        auto clf = loadModelFromFile(dir, prefix);
        if (!clf) std::cerr << "Failed to load model: " << dir << "/" << prefix << std::endl;
        return clf;
    }

    std::unordered_map<std::string, int> parseGroundTruth(const std::string& tsvPath) {
        std::unordered_map<std::string, int> groundTruth;
        std::ifstream file(tsvPath);
        
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open ground truth file: " << tsvPath << std::endl;
            return groundTruth;
        }
        
        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            size_t start = 0, end;
            std::string filename, ageStr, genderStr;
            
            for (int col = 0; col <= 6; ++col) {
                end = line.find('\t', start);
                std::string token = (end == std::string::npos)
                                  ? line.substr(start)
                                  : line.substr(start, end - start);
                
                if (col == 1) filename  = token;
                else if (col == 5) ageStr   = token;
                else if (col == 6) genderStr = token;
        
                if (end == std::string::npos) break;
                start = end + 1;
            }
            // Combine predictions to get final class (0-3)
                // 0: Male + Twenties
                // 1: Female + Twenties
                // 2: Male + Fifties
                // 3: Female + Fifties
            int ageCode    = (ageStr    == "twenties") ? 0 : 1;
            int genderCode = (genderStr == "male")    ? 0 : 1;
    
            groundTruth[filename] = ageCode * 2 + genderCode;
        }
        
        return groundTruth;
    }
};

int main(int argc, char* argv[]) {
    Inference engine(argc, argv);
    if (!engine.initialize()) return 1;
    return engine.run();
}