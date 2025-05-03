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

namespace fs = std::filesystem;

// Parse command line arguments in the format --key=value
std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Check if argument has the format --key=value
        if (arg.substr(0, 2) == "--" && arg.find('=') != std::string::npos) {
            size_t equalPos = arg.find('=');
            std::string key = arg.substr(2, equalPos - 2);
            std::string value = arg.substr(equalPos + 1);
            args[key] = value;
        }
    }
    
    return args;
}

// Sort filenames numerically (1.mp3, 2.wav, etc.)
bool numericalSort(const std::string& a, const std::string& b) {
    try {
        // Extract number prefix from filenames
        std::string numStr_a, numStr_b;
        size_t i = 0, j = 0;
        
        // Get digits from start of filename a
        while (i < a.size() && std::isdigit(a[i])) {
            numStr_a += a[i++];
        }
        
        // Get digits from start of filename b
        while (j < b.size() && std::isdigit(b[j])) {
            numStr_b += b[j++];
        }
        
        // If both have numbers, compare numerically
        if (!numStr_a.empty() && !numStr_b.empty()) {
            return std::stoi(numStr_a) < std::stoi(numStr_b);
        }
        
        // Fall back to lexicographical comparison
        return a < b;
    }
    catch (const std::exception&) {
        // If any conversion fails, fall back to string comparison
        return a < b;
    }
}

// Extract audio features using Essentia
std::vector<float> extractFeatures(const std::string& audioFile) {
    using namespace essentia;
    using namespace essentia::standard;

    AudioPreprocessor* audioProcessor = new AudioPreprocessor(4);
    std::vector<essentia::Real> audioBuffer;
    float duration = 0.0f;
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    bool success = audioProcessor->processFile(audioFile, "", duration, factory, audioBuffer, false);

    delete audioProcessor;

    if (!success) {
        std::cerr << "Error processing file: " << audioFile << std::endl;
        return {};
    }

    int numSamples = static_cast<int>(audioBuffer.size());
    if (numSamples == 0) {
        std::cerr << "No audio data in file: " << audioFile << std::endl;
        return {};
    }
    
    try {
        // Try the main feature extractor
        return getFeatureVector("", audioBuffer);
    }
    catch (const std::exception& e) {
        std::cerr << "Warning: Feature extraction error: " << e.what() << std::endl;
        std::cerr << "Using fallback feature extraction" << std::endl;
        
        // Fallback feature extraction
        std::vector<float> features;
        if (audioBuffer.size() > 100) {
            for (int i = 0; i < 100; i++) {
                int index = i * audioBuffer.size() / 100;
                features.push_back(audioBuffer[index]);
            }
        }
        return features;
    }
}

// Load a stacking classifier from the given directory
std::unique_ptr<StackingClassifier> loadModel(const std::string& modelDir, const std::string& configPrefix) {
    // Load configuration parameters from summary file
    std::ifstream summaryFile(modelDir + "/summary.txt");
    if (!summaryFile.is_open()) {
        std::cerr << "Error: Failed to open summary file" << std::endl;
        return nullptr;
    }

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
        if (line.find(prefix + "svm_c=") == 0) {
            svm_c = std::stoi(line.substr(prefix.length() + 6));
        } else if (line.find(prefix + "svm_gamma=") == 0) {
            svm_gamma = std::stod(line.substr(prefix.length() + 10));
        } else if (line.find(prefix + "rf_trees=") == 0) {
            rf_trees = std::stoi(line.substr(prefix.length() + 9));
        } else if (line.find(prefix + "knn_k=") == 0) {
            knn_k = std::stoi(line.substr(prefix.length() + 6));
        } else if (line.find(prefix + "knn_metric=") == 0) {
            knn_metric = line.substr(prefix.length() + 11);
        } else if (line.find(prefix + "n_classes=") == 0) {
            n_classes = std::stoi(line.substr(prefix.length() + 10));
        } else if (line.find(prefix + "nn_hidden1=") == 0) {
            nn_hidden1 = std::stoi(line.substr(prefix.length() + 11));
        } else if (line.find(prefix + "nn_hidden2=") == 0) {
            nn_hidden2 = std::stoi(line.substr(prefix.length() + 11));
        }
    }

    // Create base models matching training configuration
    std::vector<std::unique_ptr<BaseEstimator>> base_models;
    // base_models.push_back(std::make_unique<harmony::KNN>(knn_k, knn_metric));
    
    // Uncomment these when needed and properly implemented
    base_models.push_back(std::make_unique<harmony::SVM_ML>(svm_c, svm_gamma));
    // base_models.push_back(std::make_unique<harmony::RandomForest>(rf_trees, 5, n_classes));
    // base_models.push_back(std::make_unique<harmony::NeuralNet>(nn_hidden1, nn_hidden2, n_classes));
    
    // Create meta model
    auto meta_model = std::make_unique<harmony::LR>(0.01, n_classes);
    
    // Create stacking classifier
    auto classifier = std::make_unique<StackingClassifier>(std::move(base_models), std::move(meta_model));
    
    // Load models
    std::string modelSubdir = configPrefix.empty() ? modelDir : modelDir + "/" + configPrefix;
    if (!classifier->loadModels(modelSubdir)) {
        std::cerr << "Error: Failed to load " << configPrefix << " models from " << modelSubdir << std::endl;
        return nullptr;
    }
    
    return classifier;
}

// Parse ground truth from TSV file
std::map<std::string, int> parseGroundTruth(const std::string& tsvPath) {
    std::map<std::string, int> groundTruth;
    std::ifstream file(tsvPath);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open ground truth file: " << tsvPath << std::endl;
        return groundTruth;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item, filename;
        int label = -1;
        
        // Parse TSV line
        // input format:  client_id	path	sentence	up_votes	down_votes	age	gender	accent	label
        if (std::getline(ss, filename, '\t') && std::getline(ss, item, '\t')) {
            std::string sentence, upVotes, downVotes, ageStr, genderStr, accent, labelStr;
            if (!(std::getline(ss, sentence, '\t') &&
                std::getline(ss, upVotes, '\t') &&
                std::getline(ss, downVotes, '\t') &&
                std::getline(ss, ageStr, '\t') &&
                std::getline(ss, genderStr, '\t') &&
                std::getline(ss, accent, '\t') &&
                std::getline(ss, labelStr, '\t'))) {
                continue;
            }
            int gender = (genderStr == "male") ? 1 : 0;
            int age = (ageStr == "fifties") ? 1 : 0;
            groundTruth[filename] = gender * 2 + age;
        }
    }
    
    return groundTruth;
}

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " --data-dir=<path> --model-dir=<path> [options]\n\n"
              << "Options:\n"
              << "  --data-dir=<path>      Directory containing audio files\n"
              << "  --model-dir=<path>     Directory containing model files\n"
              << "  --ground-truth=<path>  (Optional) TSV file with ground truth labels\n"
              << "  --mode=<mode>          Mode: 'combined' for separate gender/age models,\n"
              << "                         'single' for one model (default: single)\n"
              << "  --gender-prefix=<str>  Prefix for gender model files (default: 'gender')\n"
              << "  --age-prefix=<str>     Prefix for age model files (default: 'age')\n" 
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto args = parseArgs(argc, argv);
    
    // Check required arguments
    // if (args.count("data-dir") == 0 || args.count("model-dir") == 0) {
    //     printUsage(argv[0]);
    //     return 1;
    // }
    
    // Get argument values with defaults
    std::string dataDir = "data/test";
    std::string modelDir = "models/both";
    std::string groundTruthPath = "data/datasets/filtered_data_labeled.tsv";
    std::string mode =  "single";
    std::string genderPrefix = args.count("gender-prefix") ? args["gender-prefix"] : "gender";
    std::string agePrefix = args.count("age-prefix") ? args["age-prefix"] : "age";
    
    
    // Verify directories exist
    if (!fs::exists(dataDir) || !fs::is_directory(dataDir)) {
        std::cerr << "Error: Data directory not found: " << dataDir << std::endl;
        return 1;
    }
    
    if (!fs::exists(modelDir) || !fs::is_directory(modelDir)) {
        std::cerr << "Error: Model directory not found: " << modelDir << std::endl;
        return 1;
    }
    
    // Get audio files and sort numerically
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dataDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::string extension = entry.path().extension().string();
            if (extension == ".mp3" || extension == ".wav") {
                files.push_back(filename);
            }
        }
    }
    // std::sort(files.begin(), files.end(), numericalSort);
    
    if (files.empty()) {
        std::cerr << "Error: No audio files found in directory: " << dataDir << std::endl;
        return 1;
    }
    
    // Load ground truth if provided
    std::map<std::string, int> groundTruth;
    bool evaluateAccuracy = false;
    if (!groundTruthPath.empty() && fs::exists(groundTruthPath)) {
        groundTruth = parseGroundTruth(groundTruthPath);
        evaluateAccuracy = !groundTruth.empty();
        std::cout << "Loaded " << groundTruth.size() << " ground truth labels" << std::endl;
    }
    
    // Load model(s) based on mode
    std::unique_ptr<StackingClassifier> classifier;
    std::unique_ptr<StackingClassifier> genderClassifier;
    std::unique_ptr<StackingClassifier> ageClassifier;
    
    if (mode == "combined") {
        std::cout << "Loading gender model from " << modelDir + "/" + genderPrefix << std::endl;
        genderClassifier = loadModel(modelDir, genderPrefix);
        if (!genderClassifier) {
            std::cerr << "Error: Failed to load gender model" << std::endl;
            return 1;
        }
        
        std::cout << "Loading age model from " << modelDir + "/" + agePrefix << std::endl;
        ageClassifier = loadModel(modelDir, agePrefix);
        if (!ageClassifier) {
            std::cerr << "Error: Failed to load age model" << std::endl;
            return 1;
        }
    } else {
        // Single model mode
        std::cout << "Loading Single model from " << modelDir << std::endl;
        classifier = loadModel(modelDir, "");
        if (!classifier) {
            std::cerr << "Error: Failed to load model" << std::endl;
            return 1;
        }
    }
    
    // Open output files
    std::ofstream resultsFile("results.txt");
    std::ofstream timeFile("time.txt");
    std::ofstream accuracyFile;
    
    if (evaluateAccuracy) {
        accuracyFile.open("accuracy.txt");
        if (!accuracyFile.is_open()) {
            std::cerr << "Warning: Could not open accuracy.txt for writing" << std::endl;
            evaluateAccuracy = false;
        }
    }
    
    if (!resultsFile.is_open() || !timeFile.is_open()) {
        std::cerr << "Error: Failed to open output files" << std::endl;
        return 1;
    }
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Tracking metrics if evaluating accuracy
    int correctPredictions = 0;
    int totalPredictions = 0;
    
    // Process each file
    for (const auto& file : files) {
        std::string filePath = dataDir + "/" + file;
        std::cout << "Processing " << file << std::endl;
        
        try {
            // Extract features
            std::vector<float> features = extractFeatures(filePath);
            
            if (features.empty()) {
                std::cerr << "Warning: No features extracted from " << file << std::endl;
                resultsFile << "0" << std::endl;
                continue;
            }
            
            // Convert to Eigen matrix format
            Eigen::MatrixXd X(1, features.size());
            for (size_t i = 0; i < features.size(); ++i) {
                X(0, i) = features[i];
            }
            
            int finalClass = 0;
            int gender = 0;
            int age = 0;
            
            if (mode == "combined") {
                // Make gender prediction (0=female, 1=male)
                Eigen::VectorXi genderPrediction;
                genderClassifier->predict(X, genderPrediction);
                gender = genderPrediction(0);
                
                // Make age prediction (0=thirties, 1=fifties)
                Eigen::VectorXi agePrediction;
                ageClassifier->predict(X, agePrediction);
                age = agePrediction(0);
                
                // Combine predictions to get final class (0-3)
                // 0: Female + Thirties
                // 1: Female + Fifties
                // 2: Male + Thirties
                // 3: Male + Fifties
                finalClass = gender * 2 + age;
            } else {
                // Use single model for direct prediction
                Eigen::VectorXi prediction;
                classifier->predict(X, prediction);
                finalClass = prediction(0);
            }
            
            // Write prediction to results file
            resultsFile << finalClass << std::endl;
            
            // Evaluate accuracy if ground truth is available
            if (evaluateAccuracy && groundTruth.find(file) != groundTruth.end()) {
                int trueClass = groundTruth[file];
                bool correct = (finalClass == trueClass);
                
                if (correct) {
                    correctPredictions++;
                }
                totalPredictions++;
                
                // Write detailed prediction information
                if (mode == "combined") {
                    accuracyFile << file << "\t" 
                                << "True: " << trueClass << "\t"
                                << "Pred: " << finalClass << "\t"
                                << "Gender: " << gender << "\t"
                                << "Age: " << age << "\t"
                                << (correct ? "Correct" : "Wrong") << std::endl;
                } else {
                    accuracyFile << file << "\t" 
                                << "True: " << trueClass << "\t"
                                << "Pred: " << finalClass << "\t"
                                << (correct ? "Correct" : "Wrong") << std::endl;
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing file " << file << ": " << e.what() << std::endl;
            // Output default prediction if processing fails
            resultsFile << "0" << std::endl;
        }
    }
    
    // Calculate total processing time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;
    
    // Write only the time value to time.txt
    timeFile << elapsedSeconds.count();
    
    // Write accuracy information if evaluated
    if (evaluateAccuracy && totalPredictions > 0) {
        double accuracy = static_cast<double>(correctPredictions) / totalPredictions;
        std::cout << "Accuracy: " << (accuracy * 100.0) << "% (" 
                 << correctPredictions << "/" << totalPredictions << ")" << std::endl;
        
        accuracyFile << "Overall accuracy: " << (accuracy * 100.0) << "% (" 
                     << correctPredictions << "/" << totalPredictions << ")" << std::endl;
    }
    
    // Close files
    resultsFile.close();
    timeFile.close();
    if (evaluateAccuracy) {
        accuracyFile.close();
    }
    
    std::cout << "Inference completed in " << elapsedSeconds.count() << " seconds" << std::endl;
    
    return 0;
}