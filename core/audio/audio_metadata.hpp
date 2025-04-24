#pragma once
#include <string>

class AudioMetadata {
private:
    std::string filename;
    std::string gender;
    std::string age;
    float duration; // in seconds

public:
    AudioMetadata(const std::string& filename, const std::string& gender, std::string& age, float duration = -1.0f);
    
    // Getters
    std::string getFilename() const;
    std::string getGender() const;
    std::string getAge() const;
    float getDuration() const;
    
    // Setters
    void setDuration(float duration);
};