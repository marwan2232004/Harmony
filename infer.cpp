#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Default test folder path - can be overridden by command line arg
    std::string testFolder = "data/test";
    
    // Parse command line arguments
    if (argc > 1) {
        testFolder = argv[1];
    }
    
    // Ensure test folder exists
    if (!fs::exists(testFolder)) {
        std::cerr << "Error: Test folder not found: " << testFolder << std::endl;
        return 1;
    }
    
    // Construct command to run the inference executable
    std::string command = "./bin/inference --data-dir=" + testFolder;
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Execute the inference program
    std::cout << "Executing: " << command << std::endl;
    int result = system(command.c_str());
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Write total execution time (including process startup overhead)
    std::ofstream timeFile("time.txt");
    if (timeFile.is_open()) {
        timeFile << duration.count() / 1000.0  << std::endl;
        timeFile.close();
        std::cout << "Total execution time: " << duration.count() / 1000.0 << " s" << std::endl;
    } else {
        std::cerr << "Error: Could not open total_time.txt for writing" << std::endl;
    }
    
    return result;
}