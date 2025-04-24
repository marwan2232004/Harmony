#include "audio_metadata.hpp"

AudioMetadata::AudioMetadata(const std::string& filename, const std::string& gender, std::string& age, float duration)
    : filename(filename), gender(gender), age(age), duration(duration) {}

std::string AudioMetadata::getFilename() const {
    return filename;
}

std::string AudioMetadata::getGender() const {
    return gender;
}

std::string AudioMetadata::getAge() const {
    return age;
}

float AudioMetadata::getDuration() const {
    return duration;
}

void AudioMetadata::setDuration(float duration) {
    this->duration = duration;
}
