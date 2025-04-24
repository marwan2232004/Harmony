#include "audio_util.hpp"
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <filesystem>

using namespace essentia;
using namespace standard;
namespace fs = std::filesystem;

float AudioUtil::getAudioDuration(const std::string& audioFilePath) {
    if (!fs::exists(audioFilePath)) {
        throw std::runtime_error("Audio file does not exist: " + audioFilePath);
    }
    
    essentia::init();

    try {
        // Get algorithm factory
        AlgorithmFactory& factory = AlgorithmFactory::instance();

        // Create audio loader (automatically converts to mono)
        Algorithm* audioLoader = factory.create("MonoLoader",
                                              "filename",  audioFilePath);
        
        // Buffer for audio data
        std::vector<Real> audioBuffer;
        audioLoader->output("audio").set(audioBuffer);
        
        // Compute to load the audio
        audioLoader->compute();
        
        // Get the duration (samples / sample rate)
        float duration = static_cast<float>(audioBuffer.size()) / 
                        static_cast<float>(audioLoader->parameter("sampleRate").toInt());
        
        // Clean up
        delete audioLoader;
        essentia::shutdown();
        
        return duration;
    }
    catch (const std::exception& e) {
        essentia::shutdown();
        return -1.0f; 
    }
}