#include "tonnetz.h"
#include "feature_utils.h"

using namespace essentia;
using namespace standard;

std::vector<Real> extractTonnetzFeatures(
    const std::string& filename,
    int sampleRate,
    AlgorithmFactory& factory,
    std::vector<float>& featureVector,
    bool appendToFeatureVector
) {
    std::vector<Real> audioBuffer;
    Algorithm* loader = createAudioLoader(filename, sampleRate, audioBuffer);

    Algorithm* tonal = factory.create("TonalExtractor");
    
    std::vector<std::vector<Real>> hpcpFrames;
    std::string key_key, key_scale;
    Real key_strength;
    Real chords_changes_rate;
    std::vector<Real> chords_histogram;
    std::string chords_key;
    Real chords_number_rate;
    std::vector<std::string> chords_progression;
    std::string chords_scale;
    std::vector<Real> chords_strength;
    std::vector<std::vector<Real>> hpcp_highres;  

    //most of them are useless outputs, but we need to set them to avoid errors
    tonal->input("signal").set(audioBuffer);
    tonal->output("hpcp").set(hpcpFrames);
    tonal->output("key_key").set(key_key);
    tonal->output("key_scale").set(key_scale);
    tonal->output("key_strength").set(key_strength);
    tonal->output("chords_changes_rate").set(chords_changes_rate);
    tonal->output("chords_histogram").set(chords_histogram);
    tonal->output("chords_key").set(chords_key);
    tonal->output("chords_number_rate").set(chords_number_rate);
    tonal->output("chords_progression").set(chords_progression);
    tonal->output("chords_scale").set(chords_scale);
    tonal->output("chords_strength").set(chords_strength);
    tonal->output("hpcp_highres").set(hpcp_highres);

    tonal->compute();

    std::vector<Real> features;
    
    if (!hpcpFrames.empty()) {
        std::vector<Real> hpcpAvg(12, 0.0);
        for (const auto& frame : hpcpFrames) {
            for (int i = 0; i < 12; i++) {
                hpcpAvg[i] += frame[i];
            }
        }
        for (auto& val : hpcpAvg) val /= hpcpFrames.size();
        features.insert(features.end(), hpcpAvg.begin(), hpcpAvg.end());
    }
    
    features.push_back(key_strength);

    delete loader;
    delete tonal;

    if (appendToFeatureVector) {
        featureVector.insert(featureVector.end(), features.begin(), features.end());
    }
    return features;

}