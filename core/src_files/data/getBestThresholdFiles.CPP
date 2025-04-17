#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
const int THRESHHOLD = 10000;
const int lowest_duration = 3;
const int highest_duration = 5;
const int SEED=42;

const std::string CSV_PATH = "D:/Github/Harmony/data/CSV'S/P-G-AL.csv";
const std::string CSV_PATH_COMBINED = "D:/Github/Harmony/data/CSV'S/" + std::to_string(2*THRESHHOLD) + "_SAMPLE.csv";
const std::string CSV_PATH_MALES= "D:/Github/Harmony/data/CSV'S/" + std::to_string(THRESHHOLD) + "MALES_SAMPLE.csv";
const std::string CSV_PATH_FEMALES= "D:/Github/Harmony/data/CSV'S/" + std::to_string(THRESHHOLD) + "FEMALES_SAMPLE.csv";

struct AudioFile {
    std::string path;
    std::string gender;
    int duration{};
};

std::vector<AudioFile> readCSV(const std::string& filename) {
    std::vector<AudioFile> files;
    std::ifstream in(filename);
    std::string line;

    // Skip header
    std::getline(in, line);

    while (std::getline(in, line)) {
        size_t pos1 = line.find(',');
        size_t pos2 = line.find(',', pos1 + 1);

        AudioFile file;
        file.path = line.substr(0, pos1);
        file.gender = line.substr(pos1 + 1, pos2 - pos1 - 1);
        file.duration = std::stof(line.substr(pos2 + 1));

        // Only keep files with 3-5 second duration
        if (file.duration >= lowest_duration && file.duration <= highest_duration) {
            files.push_back(file);
        }
    }

    return files;
}

void writeCSV(const std::string& filename, const std::vector<AudioFile>& files) {
    std::ofstream out(filename);
    out << "path,gender,duration\n";

    for (const auto& file : files) {
        out << file.path << "," << file.gender << "," << file.duration << "\n";
    }
}

int main() {
    // Read and filter original dataset
    auto allFiles = readCSV(CSV_PATH);

    // Separate by gender
    std::vector<AudioFile> maleFiles, femaleFiles;
    for (const auto& file : allFiles) {
        if (file.gender == "male") {
            maleFiles.push_back(file);
        } else if (file.gender == "female") {
            femaleFiles.push_back(file);
        }
    }

    // Random sampling with fixed seed for reproducibility
    std::random_device rd;
    std::mt19937 g(SEED); // Seed with 42 for consistent results

    // Shuffle and select 10K samples
    std::shuffle(maleFiles.begin(), maleFiles.end(), g);
    std::shuffle(femaleFiles.begin(), femaleFiles.end(), g);

    // Take first 10K (or max available)
    size_t sampleSize = THRESHHOLD;
    size_t maleCount = std::min(sampleSize, maleFiles.size());
    size_t femaleCount = std::min(sampleSize, femaleFiles.size());

    std::vector<AudioFile> maleSample(maleFiles.begin(), maleFiles.begin() + maleCount);
    std::vector<AudioFile> femaleSample(femaleFiles.begin(), femaleFiles.begin() + femaleCount);

    // Write to separate files
    writeCSV(CSV_PATH_MALES, maleSample);
    writeCSV(CSV_PATH_FEMALES, femaleSample);

    // Combined version (optional)
    std::vector<AudioFile> combinedSample;
    combinedSample.insert(combinedSample.end(), maleSample.begin(), maleSample.end());
    combinedSample.insert(combinedSample.end(), femaleSample.begin(), femaleSample.end());
    std::shuffle(combinedSample.begin(), combinedSample.end(), g);
    writeCSV(CSV_PATH_COMBINED, combinedSample);

    std::cout << "Created datasets:\n"
              << "- Male samples: " << maleCount << "\n"
              << "- Female samples: " << femaleCount << "\n"
              << "- Total balanced: " << combinedSample.size() << std::endl;

    return 0;
}