#include "mfcc.h"

using namespace essentia;
using namespace standard;

int main() {
    std::string path = "audio/input.wav"; // test file
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

    std::vector<essentia::Real> MFCCfeatures = extractMFCCFeatures(
        path, sampleRate, frameSize, hopSize, numberBands, numberCoefficients,
        lowFreq, highFreq, liftering, dctType, logType
    );

    std::cout << "Final feature vector (" << MFCCfeatures.size() << " dimensions):\n";
    for (auto f : MFCCfeatures) std::cout << f << " ";
    std::cout << std::endl;

    return 0;
}