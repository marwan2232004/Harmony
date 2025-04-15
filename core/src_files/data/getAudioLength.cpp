#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <unordered_map>

namespace fs = std::filesystem;

// Configuration
const std::string AUDIO_DIR = "D:/Github/NN Dataset zips";
const std::string CSV_PATH = "your_dataset.csv";
const std::string TEMP_CSV_PATH = "temp_processing.csv";
const int CHUNK_SIZE = 1000;
const int TIMEOUT_SECONDS = 5;

// Thread-safe counters and data
std::mutex mtx;
std::atomic<int> processed_count(0);
std::atomic<int> error_count(0);

struct AudioFile {
    std::string path;
    std::string gender;
    double duration;
    bool processed;
};

std::vector<AudioFile> audio_files;

double get_audio_length(const std::string& file_path) {
    try {
        // Build ffprobe command
        std::string cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"" + 
                         file_path + "\" 2>&1";
        
        // Execute command with timeout
        auto start = std::chrono::steady_clock::now();
        
        FILE* pipe = _popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to open pipe");
        }
        
        char buffer[128];
        std::string result;
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != nullptr) {
                result += buffer;
            }
            
            // Check timeout
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            if (elapsed > TIMEOUT_SECONDS) {
                _pclose(pipe);
                throw std::runtime_error("Timeout exceeded");
            }
        }
        
        int status = _pclose(pipe);
        if (status != 0) {
            throw std::runtime_error("FFprobe returned non-zero status");
        }
        
        // Clean and parse result
        result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
        result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
        
        return std::stod(result);
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cerr << "Error processing " << file_path << ": " << e.what() << std::endl;
        return 0.0;
    }
}

void process_chunk(int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx && i < audio_files.size(); i++) {
        if (audio_files[i].processed) continue;
        
        std::string full_path = AUDIO_DIR + "/" + audio_files[i].path;
        double duration = get_audio_length(full_path);
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            audio_files[i].duration = duration;
            audio_files[i].processed = true;
            processed_count++;
        }
    }
}

void save_progress() {
    std::lock_guard<std::mutex> lock(mtx);
    
    std::ofstream outfile(TEMP_CSV_PATH);
    if (!outfile.is_open()) {
        std::cerr << "Error saving progress!" << std::endl;
        return;
    }
    
    // Write header
    outfile << "path,gender,audio_length,processed\n";
    
    // Write data
    for (const auto& file : audio_files) {
        outfile << file.path << ","
                << file.gender << ","
                << file.duration << ","
                << (file.processed ? "1" : "0") << "\n";
    }
    
    outfile.close();
    
    // Replace original file
    fs::rename(TEMP_CSV_PATH, CSV_PATH);
}

void load_progress() {
    if (!fs::exists(CSV_PATH)) {
        throw std::runtime_error("Input CSV file not found");
    }
    
    std::ifstream infile(CSV_PATH);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open input CSV");
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(infile, line)) {
        if (first_line) {
            first_line = false;
            continue; // Skip header
        }
        
        std::stringstream ss(line);
        std::string token;
        AudioFile file;
        
        // Parse path
        std::getline(ss, token, ',');
        file.path = token;
        
        // Parse gender
        std::getline(ss, token, ',');
        file.gender = token;
        
        // Parse duration if exists
        if (std::getline(ss, token, ',')) {
            try {
                file.duration = std::stod(token);
                file.processed = true;
            } catch (...) {
                file.duration = 0.0;
                file.processed = false;
            }
        } else {
            file.duration = 0.0;
            file.processed = false;
        }
        
        audio_files.push_back(file);
    }
    
    infile.close();
}

int main() {
    try {
        // Load existing progress
        load_progress();
        
        int total_files = audio_files.size();
        int remaining_files = std::count_if(audio_files.begin(), audio_files.end(), 
            [](const AudioFile& f) { return !f.processed; });
        
        std::cout << "Resuming processing - " << remaining_files << " files remaining" << std::endl;
        
        // Determine number of threads
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // Fallback
        
        std::vector<std::thread> threads;
        int chunk_start = 0;
        
        auto last_save = std::chrono::steady_clock::now();
        
        while (chunk_start < audio_files.size()) {
            // Find next unprocessed chunk
            while (chunk_start < audio_files.size() && audio_files[chunk_start].processed) {
                chunk_start++;
            }
            
            if (chunk_start >= audio_files.size()) break;
            
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, (int)audio_files.size());
            
            // Process chunk with multiple threads
            int files_per_thread = (chunk_end - chunk_start) / num_threads;
            
            for (unsigned int i = 0; i < num_threads; i++) {
                int start = chunk_start + i * files_per_thread;
                int end = (i == num_threads - 1) ? chunk_end : start + files_per_thread;
                
                threads.emplace_back(process_chunk, start, end);
            }
            
            // Wait for threads to finish
            for (auto& t : threads) {
                if (t.joinable()) t.join();
            }
            threads.clear();
            
            // Save progress periodically
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_save).count() > 30) {
                save_progress();
                last_save = now;
                
                remaining_files = std::count_if(audio_files.begin(), audio_files.end(), 
                    [](const AudioFile& f) { return !f.processed; });
                
                std::cout << "Progress saved. " << remaining_files << " files remaining." << std::endl;
            }
            
            chunk_start = chunk_end;
        }
        
        // Final save
        save_progress();
        
        std::cout << "Processing complete! Processed " << processed_count << " files with " 
                  << error_count << " errors." << std::endl;
        
        // TODO: Add histogram generation (would need a plotting library)
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}