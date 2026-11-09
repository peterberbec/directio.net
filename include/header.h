#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>
#include <sqlite3.h>
#include "llama.h"
#define DB_PATH "data/directio.net.db"

#define Fit_String(s,w) (((s).size()>=(w))?(s.substr(0,(w))):((s)+std::string((w)-(s).size(),' ')))
#define GetTimestamp_LL() (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())

struct Memory;
struct LLMResponse;
class Message
{
	public:	
		std::string person;
		std::string text;
		std::int64_t time;
		int tokens;
		bool Initialize();
		~Message();
		void OutputMessage();
		std::string OutputMessage_LLM();
		void OutputMessage(Message messageT);
		void OutputMessage(std::string personT, std::int64_t timeT, std::string textT);
		void OutputHeader();
		void OutputHeader(std::string personT);
		void OutputHeader(std::string personT, std::int64_t timeT);
};
class RDSConnector
{
	private:
		sqlite3* db;
		sqlite3_stmt* stmt;
		int callback_firstcol(void* data,int argc,char** argv,char** colNames);
	public:
		sqlite3* GetDB();
		bool StoreMessage(Message inMessage);
		~RDSConnector();
		bool Initialize();
		std::vector<Message> ReturnHistory(int limit);
		void ForgetHistory();
		void PrintHistory(int limit = 20);
		std::string PrintHistory_LLM(int limit = 20);
};
class LlamaConnector
{
	private:
		llama_model* model;
		llama_context* ctx;
		llama_sampler* sampler;
		static void SilentLogger(enum ggml_log_level level,const char * text,void * user_data);
		LLMResponse ParseResponse(const std::string jsonText);
	public:
		std::vector<float> GetEmbedding(const std::string& text);
		std::string prompt(const std::string& in_prompt, RDSConnector&, Message&);
		int tokenize(const std::string& prompt);
		~LlamaConnector();
		bool InitializeM(const std::string& modelPath, bool em, int nctx);
		bool Initialize(const std::string& modelPath);
		void clearModelMemory();
};
std::string prompt_gen(const std::string user_prompt, RDSConnector&);
std::string GetTimestamp_STR(std::int64_t msin = 0);
