#include <iostream>
#include <memory>
#include <sys/stat.h>
#include "audio_util.hpp"
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <filesystem>

using namespace essentia;
using namespace standard;
namespace fs = std::filesystem;

std::vector<essentia::Real> AudioUtil::readAudioFile(const std::string& audioFilePath, float& duration, int& sampleRate) {
    if (!fs::exists(audioFilePath)) {
        throw std::runtime_error("Audio file does not exist: " + audioFilePath);
    }
    struct stat fileStat;
    // Check size
    if (stat(audioFilePath.c_str(), &fileStat) == 0 && fileStat.st_size == 0) {
        throw std::runtime_error("Audio file is empty: " + audioFilePath);
    }

    std::unique_ptr<Algorithm> audioLoader;
    try {
        // Get algorithm factory
        AlgorithmFactory& factory = AlgorithmFactory::instance();

        // Create audio loader (automatically converts to mono)
        audioLoader.reset(factory.create("MonoLoader", "filename", audioFilePath));

        // Buffer for audio data
        std::vector<Real> audioBuffer;
        audioLoader->output("audio").set(audioBuffer);
        audioLoader->compute();

        // Get the duration (samples / sample rate)
        sampleRate = audioLoader->parameter("sampleRate").toInt();
        duration = static_cast<float>(audioBuffer.size()) / 
                        static_cast<float>(sampleRate);
        
        // Clean up
        audioLoader.reset();
        return audioBuffer;
    }
    catch (const std::exception& e) {
        sampleRate = 0;
        duration = -1.0f;
        throw std::runtime_error("Error reading audio file: " + std::string(e.what()));
        return std::vector<Real>(); 
    }
}