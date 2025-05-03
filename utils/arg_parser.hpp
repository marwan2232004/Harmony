#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <locale>
#include <codecvt>

#include "logger.hpp"

namespace harmony
{

class ArgParser {
    int argc;
    char** argv;
public:
    enum OptionType { FLAG, PARAM };
    using TYPE = OptionType;
    
    struct Option {
        std::string name;
        std::string description;
        std::string defaultValue;
        OptionType type;
        bool required;
    };

    ArgParser(int argc, char* argv[]) {
        this->argc = argc;

        if (argc > 0) {
            this->argv = argv;
            programName = argv[0];
        } else {
            throw std::runtime_error("No arguments provided.");
        }

        Logger::getInstance().setupUTF8();
    }

    /**
     * @brief Adds an option to the parser.
     * 
     * @param name The name of the option (without the leading '--').
     * @param description A brief description of the option.
     * @param type The type of the option (FLAG or PARAM).
     * @param defaultValue The default value for the option (if applicable).
     * @param required Whether the option is required.
     */
    template<typename T>
    void addOption(const std::string& name,
                const std::string& description,
                T defaultValue = T(),
                OptionType type = PARAM,
                bool required = false) {
        std::stringstream ss;
        ss << defaultValue;
        options[name] = {name, description, ss.str(), type, required};
    }

    /**
     * @brief Parses command line arguments.
     * 
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    void parse() {
        std::vector<std::string> args(argv + 1, argv + argc);
        parsedValues.clear();

        for (const auto& arg : args) {
            if (arg == "--help") {
                printUsage();
                exit(0);
            }

            size_t eqPos = arg.find('=');
            std::string key = (eqPos != std::string::npos) ? arg.substr(0, eqPos) : arg;
            
            if (options.count(key.substr(2))) { 
                if (options[key.substr(2)].type == FLAG) {
                    parsedValues[key.substr(2)] = "true";
                } else {
                    parsedValues[key.substr(2)] = (eqPos != std::string::npos) ? 
                        arg.substr(eqPos + 1) : "";
                }
            }
        }

        // Set defaults for unprovided options
        for (const auto& [name, opt] : options) {
            if (!parsedValues.count(name) && !opt.defaultValue.empty()) {
                parsedValues[name] = opt.defaultValue;
            }
        }

        // Validate required options
        for (const auto& [name, opt] : options) {
            if (opt.required && !parsedValues.count(name)) {
                throw std::runtime_error("Missing required option: --" + name);
            }
        }
    }

    /**
     * @brief Retrieves the value of an option.
     * 
     * @param name The name of the option (without the leading '--').
     * @return The value of the option.
     */
    template<typename T>
    T get(const std::string& name) const {
        if (!parsedValues.count(name)) {
            throw std::runtime_error("Option not found: --" + name);
        }
        return convert<T>(parsedValues.at(name));
    }

    /**
     * @brief Retrieves the value of an option.
     * 
     * @param name The name of the option (without the leading '--').
     * @return The value of the option.
     */
    template<typename T>
    T get(const char* name) const {
        return get<T>(std::string(name));
    }

    /**
     * @brief Checks if an option was provided.
     * 
     * @param name The name of the option (without the leading '--').
     * @return True if the option was provided, false otherwise.
     */
    bool has(const std::string& name) const {
        return parsedValues.count(name) > 0;
    }

    /**
     * @brief Prints the usage information for the program.
     */
    void printUsage() const {
        std::cout << "Usage: " << programName << " [options]\n";
        std::cout << "Options:\n";
        
        size_t maxNameLen = 0;
        for (const auto& [name, opt] : options) {
            maxNameLen = std::max(maxNameLen, name.length());
        }

        for (const auto& [name, opt] : options) {
            std::string defaultValue = opt.defaultValue.empty() ? "" : 
                                      " (default: " + opt.defaultValue + ")";
            std::cout << "  --" << std::left << std::setw(maxNameLen + 2) << (name + (opt.type == PARAM ? "=<value>" : ""))
                      << " : " << opt.description << defaultValue << "\n";
        }
    }

    /**
     * @brief Prints the current configuration of the program.
     */
    void printConfig() const {
        std::cout << "⚙️  " << Logger::COLOR::YELLOW << "Model Configuration:\n" << Logger::COLOR::RESET;
        std::cout << std::string(50, '-') << "\n";
        
        size_t maxNameLen = 0;
        for (const auto& [name, _] : parsedValues) {
            maxNameLen = std::max(maxNameLen, name.length());
        }

        for (const auto& [name, value] : parsedValues) {
            std::string formattedValue = value;
            std::string formattedName = name;
            // make name capitalized and not '-' separated
            std::transform(formattedName.begin(), formattedName.end(), formattedName.begin(), ::toupper);
            std::replace(formattedName.begin(), formattedName.end(), '-', '_');
            if (options.at(name).type == FLAG) {
                formattedValue = (value == "true") ? "Enabled" : "Disabled";
            }
            
            std::cout << "▸ " << formattedName + ":"
                      << formattedValue << "\n";
        }
        std::cout << std::string(50, '-') << "\n\n";
    }

private:
    std::string programName;
    std::map<std::string, Option> options;
    std::map<std::string, std::string> parsedValues;

    /**
     * @brief Converts a string value to the specified type.
     * 
     * @tparam T The type to convert to.
     * @param value The string value to convert.
     * @return The converted value.
     */
    template<typename T>
    T convert(const std::string& value) const {
        std::istringstream iss(value);
        T result;
        iss >> result;
        return result;
    }
};

template<>
bool ArgParser::convert<bool>(const std::string& value) const {
    return (value == "true");
}

template<>
std::string ArgParser::convert<std::string>(const std::string& value) const {
    return value;
}

}