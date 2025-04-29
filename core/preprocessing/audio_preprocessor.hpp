#pragma once
#include <string>
#include <vector>
#include <essentia/essentia.h>
#include "../../tools/tqdm.cpp"  
#include "../../tools/audio_util.hpp"
#include <essentia/algorithmfactory.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>

class AudioPreprocessor {
public:
    AudioPreprocessor(float targetDuration = 5.0f);
    ~AudioPreprocessor();
    
    // Process a single file
    bool processFile(const std::string& inputPath, const std::string& outputPath, float& duration);
    
    // Process a batch of files
    int processBatch(
        const std::string& metadataPath,
        const std::string& outputDir,
        const int maxFiles,
        bool showProgress,
        int startLine = 0,
        int endLine = -1);
    
    // Configuration setters
    void enableTrimming(bool enable = true) { trimEnabled = enable; }
    void enableNormalization(bool enable = true) { normalizeEnabled = enable; }
    void enableNoiseReduction(bool enable = true) { noiseReductionEnabled = enable; }
    void enableSilenceRemoval(bool enable = true) { silenceRemovalEnabled = enable; }
    
    void setTargetDuration(float seconds) { targetDuration = seconds; }
    void setTargetRMS(float rms) { targetRMS = rms; }
    void setNoiseThreshold(float threshold) { noiseThreshold = threshold; }
    void setSilenceThreshold(float threshold) { silenceThreshold = threshold; }
    void setMinSilenceMs(int ms) { minSilenceMs = ms; }
    
private:
    // Processing parameters
    float targetDuration = 5.0f;
    float targetRMS = 0.2f;
    float noiseThreshold = 0.01f;
    float silenceThreshold = 0.01f;
    int minSilenceMs = 500;
    
    // Enabled flags
    bool trimEnabled = true;
    bool normalizeEnabled = true;
    bool noiseReductionEnabled = true;
    bool silenceRemovalEnabled = true;
    
    // Individual processing functions
    void trimAudio(std::vector<essentia::Real>& audioBuffer, int& sampleRate);
    void normalizeVolume(std::vector<essentia::Real>& audioBuffer);
    void reduceNoise(std::vector<essentia::Real>& audioBuffer);
    void removeSilence(std::vector<essentia::Real>& audioBuffer, int& sampleRate);
    
    // Utility methods
    float calculateRMS(const std::vector<essentia::Real>& buffer);
    bool writeAudioFile(const std::vector<essentia::Real>& buffer, int& sampleRate, const std::string& filePath);
};