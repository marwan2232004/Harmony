#include "mfcc.h"
#include "chroma.h"

using namespace essentia;
using namespace standard;

int main() {

    essentia::init();

    std::string path = "audio/input.wav";
    
    // MFCC parameters
    int sampleRate = 16000;
    int frameSize = 400;
    int hopSize = 160;
    int numberBands = 26;
    int numberCoefficients = 13;
    float lowFreq = 0;
    float highFreq = 8000;
    int liftering = 22;
    int dctType = 2;
    std::string logType = "dbamp";

    // Chroma parameters
    sampleRate = 16000;
    frameSize = 32768;
    hopSize = 16384;
    float minFrequency = 27.5f;
    int binsPerOctave = 36;
    float threshold = 0.0f;
    std::string normalizeType = "unit_max";
    std::string windowType = "hann";

    std::vector<essentia::Real> MFCCfeatures = extractMFCCFeatures(
        path, sampleRate, frameSize, hopSize, numberBands, numberCoefficients,
        lowFreq, highFreq, liftering, dctType, logType
    );

    std::vector<essentia::Real> ChromaFeatures = extractChromaFeatures(
        path, sampleRate, frameSize, hopSize,
        minFrequency, binsPerOctave,
        threshold, normalizeType, windowType
    );
    
    essentia::shutdown();
    return 0;
}