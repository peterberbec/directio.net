#pragma once

#include <string>

class LlamaConnector
{
public:
    bool Initialize(
        const std::string& modelPath);

    std::string Prompt(
        const std::string& prompt);

    ~LlamaConnector();

private:
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    struct llama_sampler* sampler = nullptr;
};
