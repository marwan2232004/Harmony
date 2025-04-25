#pragma once
#include <string>
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>

class AudioUtil {
public:
    static std::vector<essentia::Real> readAudioFile(const std::string& audioFilePath, float& duration, int& sampleRate);
};