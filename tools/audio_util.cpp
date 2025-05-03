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
    Algorithm* audioLoader = nullptr;
    try {
        // Get algorithm factory
        AlgorithmFactory& factory = AlgorithmFactory::instance();

        // Create audio loader (automatically converts to mono)
        audioLoader = factory.create("MonoLoader",
                                    "filename", 
                                    audioFilePath);
        
        // Buffer for audio data
        std::vector<Real> audioBuffer;
        audioLoader->output("audio").set(audioBuffer);
        
        // Compute to load the audio
        audioLoader->compute();

        sampleRate = audioLoader->parameter("sampleRate").toInt();
        
        // Get the duration (samples / sample rate)
        duration = static_cast<float>(audioBuffer.size()) / 
                        static_cast<float>(sampleRate);

        
        // Clean up
        delete audioLoader;
        
        return audioBuffer;
    }
    catch (const std::exception& e) {
        sampleRate = 0;
        duration = -1.0f;
        if(audioLoader) {
            delete audioLoader;
        }
        return std::vector<Real>(); 
    }
}