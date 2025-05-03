#include <vector>
#include "mfcc.h"
#include "chroma.h"
#include "spectral_contrast.h"
#include "tonnetz.h" 
#include "mel_spectrogram.h"
#include <essentia/algorithmfactory.h>

using namespace essentia;
using namespace standard;

std::vector<float> getFeatureVector(std::string path, std::vector<essentia::Real> inputAudio = std::vector<essentia::Real>());