#include "feature_extractor.h"
#include "feature_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <filesystem>
#include <utility>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include "../tools/tqdm.cpp"

namespace fs = std::filesystem;

// Progress bar constants
const int PROGRESS_BAR_WIDTH = 50;
const std::string COLOR_GREEN = "\033[32m";
const std::string COLOR_RED = "\033[31m";
const std::string COLOR_RESET = "\033[0m";

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input-metadata=<path>  Path to cleaned metadata TSV (default: data/processed/metadata_balanced.tsv)" << std::endl;
    std::cout << "  --dataset-path=<path>    Base directory for audio files (default: data/processed)" << std::endl;
    std::cout << "  --output-dir=<path>      Output directory for TSV files (default: data/features)" << std::endl;
    std::cout << "  --test-ratio=<ratio>     Test data ratio (0.0-1.0, default: 0.2)" << std::endl;
    std::cout << "  --random-seed=<seed>     Random seed for shuffling (optional)" << std::endl;
    std::cout << "  --help                   Display this help message" << std::endl;
}

std::string getParamValue(const std::string& arg, const std::string& paramName) {
    std::string prefix = "--" + paramName + "=";
    if (arg.substr(0, prefix.size()) == prefix) {
        return arg.substr(prefix.size());
    }
    return "";
}

std::vector<std::string> getFeatureNames() {
    std::vector<std::string> featureNames;
    
    // MFCC features (13 coefficients + 13 stddev)
    // for (int i = 1; i <= 26; i++) {
    //     featureNames.push_back("mfcc_mean_" + std::to_string(i));
    // }
    // for (int i = 1; i <= 26; i++) {
    //     featureNames.push_back("mfcc_std_" + std::to_string(i));
    // }
    
    // // Chroma features (36 bins + 36 stddev)
    // for (int i = 1; i <= 36; i++) {
    //     featureNames.push_back("chroma_mean_" + std::to_string(i));
    // }
    // for (int i = 1; i <= 36; i++) {
    //     featureNames.push_back("chroma_std_" + std::to_string(i));
    // }
    
    // // Spectral Contrast features (6 peaks + 6 valleys + their stddevs)
    // for (int i = 1; i <= 6; i++) {
    //     featureNames.push_back("spectral_peak_mean_" + std::to_string(i));
    // }
    // for (int i = 1; i <= 6; i++) {
    //     featureNames.push_back("spectral_valley_mean_" + std::to_string(i));
    // }
    // for (int i = 1; i <= 6; i++) {
    //     featureNames.push_back("spectral_peak_std_" + std::to_string(i));
    // }
    // for (int i = 1; i <= 6; i++) {
    //     featureNames.push_back("spectral_valley_std_" + std::to_string(i));
    // }
    
    // // Tonnetz features (12 HPCP + 1 key strength)
    // for (int i = 1; i <= 12; i++) {
    //     featureNames.push_back("tonnetz_hpcp_" + std::to_string(i));
    // }
    // featureNames.push_back("tonnetz_key_strength");
    
    // Mel Spectrogram features (40 bands + 40 stddev)
    for (int i = 1; i <= 40; i++) {
        featureNames.push_back("mel_mean_" + std::to_string(i));
    }
    for (int i = 1; i <= 40; i++) {
        featureNames.push_back("mel_std_" + std::to_string(i));
    }
    
    return featureNames;
}

void printColored(const std::string& message, const std::string& color) {
    std::cout << color << message << COLOR_RESET << std::endl;
}

int main(int argc, char* argv[]) {
    // Record start time
    auto programStart = std::chrono::high_resolution_clock::now();

    // Default parameters
    std::string inputMetadata = "data/processed/metadata_balanced.tsv";
    std::string datasetPath = "data/processed";
    std::string outputDir = "data/features";
    float testRatio = 0.2f;
    int randomSeed = -1;

    // Parse command line arguments
    std::vector<std::string> args(argv + 1, argv + argc);
    for (const auto& arg : args) {
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::string value;
            if (!(value = getParamValue(arg, "input-metadata")).empty()) {
                inputMetadata = value;
            } else if (!(value = getParamValue(arg, "dataset-path")).empty()) {
                datasetPath = value;
            } else if (!(value = getParamValue(arg, "output-dir")).empty()) {
                outputDir = value;
            } else if (!(value = getParamValue(arg, "test-ratio")).empty()) {
                testRatio = std::stof(value);
                if (testRatio < 0 || testRatio > 1) {
                    std::cerr << "Error: test-ratio must be between 0.0 and 1.0\n";
                    return 1;
                }
            } else if (!(value = getParamValue(arg, "random-seed")).empty()) {
                randomSeed = std::stoi(value);
            }
        }
    }

    // Print startup information
    std::cout << "\n✨ " << COLOR_GREEN << "Starting Feature Extraction Pipeline" << COLOR_RESET << " ✨\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << "⚙️  Configuration Parameters:\n";
    std::cout << "▸ Input Metadata:    " << inputMetadata << "\n";
    std::cout << "▸ Dataset Path:      " << datasetPath << "\n";
    std::cout << "▸ Output Directory:  " << outputDir << "\n";
    std::cout << "▸ Test Split Ratio:  " << testRatio << "\n";
    std::cout << "▸ Random Seed:       " << (randomSeed == -1 ? "System Random" : std::to_string(randomSeed)) << "\n";
    std::cout << std::string(50, '-') << "\n\n";

    // Validate input metadata
    if (!fs::exists(inputMetadata)) {
        printColored("❌ Error: Input metadata file not found: " + inputMetadata, COLOR_RED);
        return 1;
    }

    // Create output directory
    fs::create_directories(outputDir);

    // Read metadata entries
    std::vector<std::tuple<std::string, std::string, std::string>> samples;
    std::ifstream file(inputMetadata);
    std::string line;
    bool firstLine = true;
    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }
        std::istringstream iss(line);
        std::string path, ageLabel, genderLabel;
        if (std::getline(iss, path, '\t') && std::getline(iss, ageLabel, '\t') && std::getline(iss, genderLabel, '\t')) {
            samples.emplace_back(path, ageLabel, genderLabel);
        }
    }

    if (samples.empty()) {
        printColored("❌ Error: No valid samples in metadata file", COLOR_RED);
        return 1;
    }

    // Shuffle samples
    if (randomSeed != -1) {
        std::mt19937 rng(randomSeed);
        std::shuffle(samples.begin(), samples.end(), rng);
    } else {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(samples.begin(), samples.end(), rng);
    }

    // Split into train and test
    size_t splitIdx = samples.size() * (1.0f - testRatio);
    auto trainSamples = std::vector<std::tuple<std::string, std::string, std::string>>(
        samples.begin(), samples.begin() + splitIdx
    );
    auto testSamples = std::vector<std::tuple<std::string, std::string, std::string>>(
        samples.begin() + splitIdx, samples.end()
    );

    // Initialize Essentia
    initializeEssentia();

    // Process batches with progress tracking
    auto processBatch = [&](const auto& batch, const std::string& filename) -> std::pair<int, int> {
        std::ofstream out(fs::path(outputDir) / filename);
        int successCount = 0;
        int errorCount = 0;
        const size_t totalFiles = batch.size();
        
        auto batchStart = std::chrono::high_resolution_clock::now();
        
        auto featureNames = getFeatureNames();
        for (const auto& name : featureNames) {
            out << name << "\t";
        }
        out << "age\tgender\n";

        Tqdm tqdm(totalFiles, "🚀 Processing " + filename + " (" + std::to_string(totalFiles) + " files)");

        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& [relPath, ageLabel, genderLabel] = batch[i];
            fs::path fullPath = fs::path(datasetPath) / relPath;
    
            try {
                if (!fs::exists(fullPath)) {
                    throw std::runtime_error("File not found");
                }
    
                std::vector<float> features = getFeatureVector(fullPath.string());
                for (const auto& feature : features) {
                    out << feature << "\t";
                }
                out << ageLabel << "\t" << genderLabel << "\n";
                successCount++;
            } catch (const std::runtime_error& e) {
                std::cerr << "\nError processing " << fullPath << ": " << e.what() << "\n";
                errorCount++;
            }
    
            tqdm.update();
        }
    
        tqdm.finish();

        auto batchEnd = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(batchEnd - batchStart);

        std::cout << "\n\n✅ Batch completed in " << duration.count() << "s\n";
        std::cout << "   Success: " << COLOR_GREEN << successCount << COLOR_RESET;
        std::cout << " | Errors: " << (errorCount > 0 ? COLOR_RED : "") << errorCount << COLOR_RESET << "\n";

        return {successCount, errorCount};
    };

    // Process both splits
    auto [trainSuccess, trainErrors] = processBatch(trainSamples, "train.tsv");
    auto [testSuccess, testErrors] = processBatch(testSamples, "test.tsv");

    // Shutdown Essentia
    shutdownEssentia();

    // Calculate total statistics
    int totalSuccess = trainSuccess + testSuccess;
    int totalErrors = trainErrors + testErrors;
    auto programEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(programEnd - programStart);

    // Final report
    std::cout << "\n" << std::string(50, '=') << "\n";
    printColored("🎉 Feature Extraction Complete!", COLOR_GREEN);
    std::cout << "⏱️  Total Time:      " << totalDuration.count() << " seconds\n";
    std::cout << "📊 Total Processed: " << (totalSuccess + totalErrors) << " files\n";
    std::cout << "✅ Successful:      " << COLOR_GREEN << totalSuccess << COLOR_RESET << "\n";
    std::cout << "❌ Failed:          " << (totalErrors > 0 ? COLOR_RED : "") << totalErrors << COLOR_RESET << "\n";
    std::cout << "📂 Output Files:\n";
    std::cout << "   - " << (fs::path(outputDir) / "train.tsv").string() << "\n";
    std::cout << "   - " << (fs::path(outputDir) / "test.tsv").string() << "\n";
    std::cout << std::string(50, '=') << "\n\n";

    return (totalErrors == 0) ? 0 : 1;
}