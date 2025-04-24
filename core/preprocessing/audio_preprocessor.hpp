#pragma once
#include <string>
#include <vector>
#include <essentia/essentia.h>

class AudioPreprocessor {
public:
    AudioPreprocessor(float targetDuration = 5.0f);
    ~AudioPreprocessor();
    
    // Process a single file
    bool processFile(const std::string& inputPath, const std::string& outputPath);
    
    // Process a batch of files
    std::vector<std::string> processBatch(
        const std::vector<std::string>& inputPaths,
        const std::string& outputDir,
        bool showProgress = true);
    
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
    void trimAudio(std::vector<essentia::Real>& audioBuffer, int sampleRate);
    void normalizeVolume(std::vector<essentia::Real>& audioBuffer);
    void reduceNoise(std::vector<essentia::Real>& audioBuffer);
    void removeSilence(std::vector<essentia::Real>& audioBuffer, int sampleRate);
    
    // Utility methods
    float calculateRMS(const std::vector<essentia::Real>& buffer);
    std::vector<essentia::Real> readAudioFile(const std::string& filePath, int& sampleRate);
    bool writeAudioFile(const std::vector<essentia::Real>& buffer, int sampleRate, const std::string& filePath);
};