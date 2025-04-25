#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include "cleaner.hpp"
#include "../audio/audio_metadata.hpp"
#include "../tools/audio_util.hpp"
#include "../tools/tqdm.cpp"

class DatasetCleaner {
private:
    std::string datasetPath;
    std::string metadataFilePath;
    std::vector<AudioMetadata> allMetadata;
    std::map<std::string, std::vector<AudioMetadata>> categorizedMetadata;
    
    // Configuration
    std::vector<std::string> genders = {"male", "female"};
    std::vector<std::string> ageGroups = {"twenties", "fifties"};
    int samplesPerCategory = 100; // default 100 samples per category
    
    // Helper methods
    void loadMetadata();
    void categorizeMetadata();
    std::string getCategoryKey(const std::string& gender, const std::string& ageGroup) const;
    bool isWithinDurationRange(float duration) const;

public:
    DatasetCleaner(
        const std::string& datasetPath, 
        const std::string& metadataFilePath
    );

    void cleanMetadata();

    void clean(bool cleanMetadata);
    void exportCleanedDataset(const std::string& outputMetadataPath) const;
    
    // Setters for configuration
    void setSamplesPerCategory(int samples);
    void setGenders(const std::vector<std::string>& genders);
    void setAgeGroups(const std::vector<std::string>& ageGroups);
};