#include "../core/cleaning/cleaner.hpp"
#include <iostream>


void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --dataset-path=<path>         Path to the dataset directory" << std::endl;
    std::cout << "  --metadata-file=<file>       Path to the metadata file" << std::endl;
    std::cout << "  --target-duration=<seconds>   Target duration for audio files" << std::endl;
    std::cout << "  --duration-tolerance=<seconds> Duration tolerance for audio files" << std::endl;
    std::cout << "  --samples-per-category=<num>  Number of samples per category" << std::endl;
    std::cout << "  --clean-tsv            Clean TSV file" << std::endl;
    std::cout << "  --help                        Display this help message" << std::endl;
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
    std::string datasetPath = "data/datasets";
    std::string metadataFile = "data/datasets/filtered_data_labeled.tsv";
    float targetDuration = 5.0f;  // 5 seconds
    float durationTolerance = 1.0f;  // ±1 second
    int samplesPerCategory = 500;
    bool cleanTSV = false;

    // Process command line arguments
    std::vector<std::string> args(argv + 1, argv + argc);
    for (const auto& arg : args) {
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if(arg == "--clean-tsv"){
            cleanTSV = true;
        } 
        else {
            // Parse parameters with values
            std::string value;

            if (!(value = getParamValue(arg, "dataset-path")).empty()) {
                datasetPath = value;
            } else if (!(value = getParamValue(arg, "metadata-file")).empty()) {
                metadataFile = value;
            } else if (!(value = getParamValue(arg, "target-duration")).empty()) {
                targetDuration = std::stof(value);
            } else if (!(value = getParamValue(arg, "duration-tolerance")).empty()) {
                durationTolerance = std::stof(value);
            } else if (!(value = getParamValue(arg, "samples-per-category")).empty()) {
                samplesPerCategory = std::stoi(value);
            }
        }
    }
    
    
    try {
        std::cout << "Starting dataset cleaning process..." << std::endl;
        std::cout << "Dataset path: " << datasetPath << std::endl;
        std::cout << "Metadata file: " << metadataFile << std::endl;
        std::cout << "Target duration: " << targetDuration << " seconds (±" << durationTolerance << ")" << std::endl;
        
        // Create and configure dataset cleaner
        DatasetCleaner cleaner(datasetPath, metadataFile, targetDuration, durationTolerance, samplesPerCategory);
        
        // Set specific age groups and genders we want
        cleaner.setAgeGroups({"twenties", "fifties"});
        cleaner.setGenders({"male", "female"});
        
        // Clean dataset
        std::cout << "Loading and filtering audio metadata..." << std::endl;
        cleaner.clean(cleanTSV);
        
        // Export cleaned dataset
        std::string outputMetadataPath = datasetPath + "/metadata_balanced.tsv";
        cleaner.exportCleanedDataset(outputMetadataPath);
        
        std::cout << "Dataset balancing completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}