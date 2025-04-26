#include "../core/preprocessing/audio_preprocessor.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --tsv-file=<path>      : TSV file containing audio file paths (default: data/metadata.tsv)" << std::endl;
    std::cout << "  --output-dir=<path>    : Output directory for processed files (default: data/processed)" << std::endl;
    std::cout << "  --max-files=<num>      : Maximum number of files to process (default: 15000)" << std::endl;
    std::cout << "  --start-line=<num>     : Start processing from this line (default: 0)" << std::endl;
    std::cout << "  --end-line=<num>       : Stop processing at this line (default: -1, process all)" << std::endl;
    std::cout << "  --target-duration=<sec>: Target duration in seconds (default: 5.0)" << std::endl;
    std::cout << "  --target-rms=<level>   : Target RMS level (0.0-1.0) (default: 0.2)" << std::endl;
    std::cout << "  --noise-threshold=<lvl>: Noise threshold (default: 0.01)" << std::endl;
    std::cout << "  --silence-threshold=<s>: Silence threshold (default: 0.01)" << std::endl;
    std::cout << "  --min-silence-ms=<ms>  : Minimum silence duration in ms (default: 500)" << std::endl;
    std::cout << "  --no-trim              : Disable trimming" << std::endl;
    std::cout << "  --no-normalize         : Disable volume normalization" << std::endl;
    std::cout << "  --no-noise-reduction   : Disable noise reduction" << std::endl;
    std::cout << "  --no-silence-removal   : Disable silence removal" << std::endl;
    std::cout << "  --help                 : Display this help message" << std::endl;
    std::cout << "TSV Format:" << std::endl;
    std::cout << "  The first column should contain the path to the audio file." << std::endl;
    std::cout << "  Other columns are optional and will be ignored." << std::endl;
    std::cout << "  Example: path/to/audio.wav\tage\tgender\tduration" << std::endl;
    std::cout << "Batch Processing:" << std::endl;
    std::cout << "  To process a large dataset in chunks, use --start-line and --end-line." << std::endl;
    std::cout << "  Example: process 1000 files at a time:" << std::endl;
    std::cout << "  ./process_dataset --start-line=0 --end-line=1000" << std::endl;
    std::cout << "  ./process_dataset --start-line=1000 --end-line=2000" << std::endl;
}

// Parse a command line parameter in the format --param=value
std::string getParamValue(const std::string& arg, const std::string& paramName) {
    std::string prefix = "--" + paramName + "=";
    if (arg.substr(0, prefix.length()) == prefix) {
        return arg.substr(prefix.length());
    }
    return "";
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string tsvFile = "data/datasets/filtered_data_labeled.tsv";
    std::string outputDir = "data/processed";
    float targetDuration = 5.0f;
    float targetRMS = 0.2f;
    float noiseThreshold = 0.01f;
    float silenceThreshold = 0.01f;
    int minSilenceMs = 500;
    int maxFiles = 15000;
    int startLine = 0;
    int endLine = -1;
    
    bool enableTrim = true;
    bool enableNormalize = true;
    bool enableNoiseReduction = true;
    bool enableSilenceRemoval = true;
    
    // Process command line arguments
    std::vector<std::string> args(argv + 1, argv + argc);
    
    for (const auto& arg : args) {
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--no-trim") {
            enableTrim = false;
        } else if (arg == "--no-normalize") {
            enableNormalize = false;
        } else if (arg == "--no-noise-reduction") {
            enableNoiseReduction = false;
        } else if (arg == "--no-silence-removal") {
            enableSilenceRemoval = false;
        } else {
            // Parse parameters with values
            std::string value;
            
            if (!(value = getParamValue(arg, "tsv-file")).empty()) {
                tsvFile = value;
            } else if (!(value = getParamValue(arg, "output-dir")).empty()) {
                outputDir = value;
            } else if (!(value = getParamValue(arg, "target-duration")).empty()) {
                targetDuration = std::stof(value);
            } else if (!(value = getParamValue(arg, "target-rms")).empty()) {
                targetRMS = std::stof(value);
            } else if (!(value = getParamValue(arg, "noise-threshold")).empty()) {
                noiseThreshold = std::stof(value);
            } else if (!(value = getParamValue(arg, "silence-threshold")).empty()) {
                silenceThreshold = std::stof(value);
            } else if (!(value = getParamValue(arg, "min-silence-ms")).empty()) {
                minSilenceMs = std::stoi(value);
            } else if (!(value = getParamValue(arg, "max-files")).empty()) {
                maxFiles = std::stoi(value);
            } else if (!(value = getParamValue(arg, "start-line")).empty()) {
                startLine = std::stoi(value);
            } else if (!(value = getParamValue(arg, "end-line")).empty()) {
                endLine = std::stoi(value);
            }
        }
    }
    
    // Display configuration
    std::cout << "=== Audio Dataset Processor ===" << std::endl;
    std::cout << "TSV file:           " << tsvFile << std::endl;
    std::cout << "Output directory:   " << outputDir << std::endl;
    std::cout << "Max files:          " << maxFiles << std::endl;
    std::cout << "Processing range:   " << startLine << " to " << (endLine == -1 ? "end" : std::to_string(endLine)) << std::endl;
    std::cout << "Target duration:    " << targetDuration << " seconds" << std::endl;
    std::cout << "Target RMS:         " << targetRMS << std::endl;
    std::cout << "Noise threshold:    " << noiseThreshold << std::endl;
    std::cout << "Silence threshold:  " << silenceThreshold << std::endl;
    std::cout << "Min silence:        " << minSilenceMs << " ms" << std::endl;
    std::cout << "Processing steps:" << std::endl;
    std::cout << "- Trimming:         " << (enableTrim ? "Enabled" : "Disabled") << std::endl;
    std::cout << "- Normalization:    " << (enableNormalize ? "Enabled" : "Disabled") << std::endl;
    std::cout << "- Noise reduction:  " << (enableNoiseReduction ? "Enabled" : "Disabled") << std::endl;
    std::cout << "- Silence removal:  " << (enableSilenceRemoval ? "Enabled" : "Disabled") << std::endl;
    
    // Check if TSV file exists
    if (!fs::exists(tsvFile)) {
        std::cerr << "Error: TSV file does not exist: " << tsvFile << std::endl;
        return 1;
    }
    
    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        std::cout << "Creating output directory: " << outputDir << std::endl;
        fs::create_directories(outputDir);
    }
    

    // Initialize audio preprocessor with configuration
    AudioPreprocessor preprocessor(targetDuration);
    
    // Configure processor
    preprocessor.enableTrimming(enableTrim);
    preprocessor.enableNormalization(enableNormalize);
    preprocessor.enableNoiseReduction(enableNoiseReduction);
    preprocessor.enableSilenceRemoval(enableSilenceRemoval);
    
    preprocessor.setTargetDuration(targetDuration);
    preprocessor.setTargetRMS(targetRMS);
    preprocessor.setNoiseThreshold(noiseThreshold);
    preprocessor.setSilenceThreshold(silenceThreshold);
    preprocessor.setMinSilenceMs(minSilenceMs);
    
    // Process files with progress bar
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> processedFiles = preprocessor.processBatch(
        tsvFile, outputDir, maxFiles, true, startLine, endLine);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    

    int nextStartLine = (endLine == -1) ? startLine + maxFiles : endLine;
    
    // Final report
    std::cout << "\nProcessing Results:" << std::endl;
    std::cout << "Time taken: " << duration << " seconds" << std::endl;
    std::cout << "Processed files saved to: " << outputDir << std::endl;
    std::cout << "Next batch should start at line: " << nextStartLine << std::endl;
    std::cout << "To continue processing, run:" << std::endl;
    std::cout << "./process_dataset --start-line=" << nextStartLine << " --end-line=" << (nextStartLine + maxFiles) << std::endl;
    
    return 0;
}