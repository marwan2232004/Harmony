#include "mfcc.h"
#include "chroma.h"
#include "spectral_contrast.h"
#include "tonnetz.h" 
#include "mel_spectrogram.h"

using namespace essentia;
using namespace standard;

int main() {
    essentia::init();

    std::string path = "audio/input.wav";
    int sampleRate = 16000;
    
    std::vector<essentia::Real> MFCCfeatures = extractMFCCFeatures(
        path, sampleRate, 400, 160, 26, 13, 0, 8000, 22, 2, "dbamp"
    );

    std::vector<essentia::Real> ChromaFeatures = extractChromaFeatures(
        path, sampleRate, 32768, 16384, 27.5f, 36, 0.0f, "unit_max", "hann"
    );

    std::vector<essentia::Real> SpectralContrastFeatures = extractSpectralContrastFeatures(
        path, sampleRate, 2048, 1024, 6, 20, 8000, 0.4f, 1.0f
    );

    std::vector<essentia::Real> TonnetzFeatures = extractTonnetzFeatures(
        path, sampleRate
    );

    std::vector<essentia::Real> MelSpectrogramFeatures = extractMelSpectrogramFeatures(
        path, sampleRate, 2048, 1024, 40, 20, 8000, "htkMel", "linear", "unit_sum", "power"
    );
    
    essentia::shutdown();
    return 0;
}