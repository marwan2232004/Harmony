#include "../core/cleaning/cleaner.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    // Default parameters
    std::string datasetPath = "data/datasets";
    std::string metadataFile = "data/datasets/filtered_data_labeled.tsv";
    float targetDuration = 5.0f;  // 5 seconds
    float durationTolerance = 0.0f;  // ±1 second
    int samplesPerCategory = 500;
    bool cleanTSV = true;
    
    // Override with command line arguments if provided
    if (argc > 1) datasetPath = argv[1];
    if (argc > 2) metadataFile = argv[2];
    if (argc > 3) targetDuration = std::stof(argv[3]);
    if (argc > 4) durationTolerance = std::stof(argv[4]);
    if (argc > 5) samplesPerCategory = std::stoi(argv[5]);
    if (argc > 6) cleanTSV = std::stoi(argv[6]);
    
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