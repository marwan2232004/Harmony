#include "cleaner.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <filesystem>

#define MAX_ROWS 15000

namespace fs = std::filesystem;

DatasetCleaner::DatasetCleaner(
    const std::string &datasetPath,
    const std::string &metadataFilePath,
    int samplesPerCategory) : datasetPath(datasetPath),
                              metadataFilePath(metadataFilePath),
                              samplesPerCategory(samplesPerCategory) {}

void DatasetCleaner::cleanMetadata()
{
    // Load tsv file and drop unnecessary columns also remove rows with files that do not exist
    std::ifstream file(metadataFilePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open metadata file: " + metadataFilePath);
    }

    // Count number of lines for progress bar
    size_t lineCount = 0;
    std::string line;
    while (std::getline(file, line))
    {
        lineCount++;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    // Initialize progress bar
    Tqdm tqdm(lineCount, "Cleaning metadata");
    std::vector<std::string> cleanedMetadata;

    int cnt = 0;

    while (std::getline(file, line))
    {
        if(cnt >= MAX_ROWS)
        {
            break;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Split by tab
        while (std::getline(ss, token, '\t'))
        {
            tokens.push_back(token);
        }

        // Check if enough tokens exist
        if (tokens.size() < 7)
        {
            tqdm.update();
            continue;
        }

        // Check if the file exists AND can be opened
        std::string filename = tokens[1];
        std::string fullPath = datasetPath + "/" + filename;

        float duration = -1.0f;
        int sampleRate = 0;
        if (fs::exists(fullPath))
        {
           AudioUtil::readAudioFile(fullPath, duration, sampleRate);
        }

        if (duration > 0)
        {
            std::string cleanedLine = tokens[1] + "\t" + tokens[5] + "\t" + tokens[6] + "\t" + std::to_string(duration);
            cleanedMetadata.push_back(cleanedLine);
            cnt++;
        }

        tqdm.update();
    }

    file.close();
    tqdm.finish();

    // Save cleaned metadata to a new file
    std::string cleanedMetadataFilePath = datasetPath + "/cleaned_metadata.tsv";
    std::ofstream cleanedFile(cleanedMetadataFilePath);
    if (!cleanedFile.is_open())
    {
        throw std::runtime_error("Could not create cleaned metadata file: " + cleanedMetadataFilePath);
    }

    // Write header
    cleanedFile << "path\tage\tgender\tduration\n";
    Tqdm saveProgress(cleanedMetadata.size(), "Saving cleaned metadata");
    for (const auto &metadata : cleanedMetadata)
    {
        cleanedFile << metadata << "\n";
        saveProgress.update();
    }

    cleanedFile.close();
    saveProgress.finish();

    // Set the metadataFilePath to the new file
    metadataFilePath = cleanedMetadataFilePath;
    std::cout << "Cleaned metadata saved to: " << cleanedMetadataFilePath << std::endl;
    std::cout << "Kept " << cleanedMetadata.size() << " valid files out of " << lineCount << " total entries" << std::endl;
}

void DatasetCleaner::loadMetadata()
{
    std::ifstream file(metadataFilePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open metadata file: " + metadataFilePath);
    }

    // Count number of lines for progress bar
    size_t lineCount = 0;
    std::string line;
    while (std::getline(file, line))
    {
        lineCount++;
    }
    file.clear();
    file.seekg(0, std::ios::beg);

    // Initialize progress bar (subtract 1 for header)
    Tqdm tqdm(lineCount - 1, "Loading metadata");

    // Assuming first line is header, skip it
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Split by tab
        while (std::getline(ss, token, '\t'))
        {
            tokens.push_back(token);
        }

        //  format:path	age	gender duration
        if (tokens.size() >= 3)
        {
            std::string filename = tokens[0];
            std::string age = tokens[1];
            std::string gender = tokens[2];
            float duration = std::stof(tokens[3]);

            // Create AudioMetadata object
            AudioMetadata metadata(filename, gender, age, duration);

            allMetadata.push_back(metadata);
        }
        tqdm.update();
    }
    file.close();
    tqdm.finish();
}

void DatasetCleaner::categorizeMetadata()
{
    categorizedMetadata.clear();

    Tqdm tqdm(allMetadata.size(), "Categorizing metadata");

    for (const auto &metadata : allMetadata)
    {
        // Categorize by gender and age group
        for (const auto &gender : genders)
        {
            if (metadata.getGender() == gender)
            {
                for (const auto &ageGroup : ageGroups)
                {
                    if (metadata.getAge() == ageGroup)
                    {
                        std::string categoryKey = getCategoryKey(gender, ageGroup);
                        categorizedMetadata[categoryKey].push_back(metadata);
                        break; // Each audio belongs to only one age group
                    }
                }
            }
        }
        tqdm.update();
    }
    tqdm.finish();

    std::cout << "\nCategorized metadata into " << categorizedMetadata.size() << " categories" << std::endl;
}

std::string DatasetCleaner::getCategoryKey(const std::string &gender, const std::string &ageGroup) const
{
    return gender + "_" + ageGroup;
}


void DatasetCleaner::clean(bool cleanMetadata)
{
    if (cleanMetadata)
    {
        DatasetCleaner::cleanMetadata();
    }
    // Load metadata from TSV file
    loadMetadata();

    // Categorize metadata by gender and age group
    categorizeMetadata();
}

void DatasetCleaner::exportCleanedDataset(const std::string &outputMetadataPath) const
{
    // Open output metadata file
    std::ofstream outFile(outputMetadataPath);
    if (!outFile.is_open())
    {
        throw std::runtime_error("Could not create output metadata file: " + outputMetadataPath);
    }

    // Write header
    outFile << "filename\tage\tgender\tduration\n";

    // Find the minimum number of samples across all categories for balance
    int minSamples = samplesPerCategory;
    for (const auto &[category, metadataList] : categorizedMetadata)
    {
        minSamples = std::min(minSamples, static_cast<int>(metadataList.size()));
        std::cout << "Category " << category << " has " << metadataList.size() << " samples" << std::endl;
    }

    std::cout << "Using " << minSamples << " samples per category for balance" << std::endl;

    Tqdm tqdm(categorizedMetadata.size(), "Exporting cleaned dataset");

    // Process each category
    for (const auto &[category, metadataList] : categorizedMetadata)
    {
        std::vector<AudioMetadata> selectedMetadata = metadataList;

        // Shuffle the metadata to get a random sample
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(selectedMetadata.begin(), selectedMetadata.end(), g);

        // Take only the required number of samples
        int numSamples = std::min(static_cast<int>(selectedMetadata.size()), minSamples);

        // Copy files and write metadata
        for (int i = 0; i < numSamples; ++i)
        {
            const auto &metadata = selectedMetadata[i];
            // Write metadata to output file
            outFile << metadata.getFilename() << "\t"
                    << metadata.getAge() << "\t"
                    << metadata.getGender() << "\t"
                    << metadata.getDuration() << "\n";
        }
        tqdm.update();
    }

    outFile.close();
    tqdm.finish();
}


void DatasetCleaner::setSamplesPerCategory(int samples)
{
    samplesPerCategory = samples;
}

void DatasetCleaner::setGenders(const std::vector<std::string> &genders)
{
    this->genders = genders;
}

void DatasetCleaner::setAgeGroups(const std::vector<std::string> &ageGroups)
{
    this->ageGroups = ageGroups;
}