#pragma once
#include <string>
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>

class AudioUtil {
public:
    static float getAudioDuration(const std::string& audioFilePath);
};