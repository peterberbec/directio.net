#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include "llama.h"

//prompt_gen.cpp

struct Memory
{
	std::string memory;
	std::string category;
	int importance;
	int64_t created_at;
};

struct LLMResponse
{
	std::string answer;
	std::string memory;
	int importance;
	std::string category;
};

std::string prompt_gen(const std::string user_prompt, RDSConnector& myRDS)
{
	std::stringstream ss;
	ss <<	std::string(
		"<|im_start|>system\n") +
		"You must format your response in valid JSON.\n"+
		"For every user message:\n" +
		"1. Answer the user.\n" +
		"2. Determine whether the message contains a memory.\n" +
		"A memory is:\n" +
		"- a preference\n" +
		"- a personal fact\n" +
		"- a goal\n" +
		"- a relationship\n" +
		"- a recurring interest\n" +
		"If a memory exists:\n" +
		"- your answer MUST be stored in the answer field.\n" +
		"- the memory MUST be stored in the memory field.\n" +
		"- importance MUST be between 1-10 and stored in the imporance fied.\n" +
		"- category MUST be stored in the category field.\n" +
		"If this should not be a memory:\n" +
		"- you MUST stored your answer in the answer field.\n" +
		"- you MUST store \"\" in the memory fied.\n" +
		"- you MUST store 0 in the importance field.\n" +
		"- you MUST store \"\" in the category field.\n" +
		"User: Do I like Audo?\n" +
		"LLM:\n" +
		"{\n" +
		"  \"answer\":\"4\",\n" +
		"  \"memory\":\"\",\n" +
		"  \"importance\":0,\n" +
		"  \"category\":\"\"\n" +
		"}\n" +
		"User: My name is Peter.\n" +
		"LLM:\n" +
		"{\n" +
		"  \"answer\":\"Nice to meet you.\",\n" +
		"  \"memory\":\"User's name is Peter\",\n" +
		"  \"importance\":10,\n" +
		"  \"category\":\"identity\"\n" +
		"}\n" +
		"User: I like German cars.\n" +
		"LLM:\n" +
		"{\n" +
		"  \"answer\":\"Interesting.\",\n" +
		"  \"memory\":\"User likes German cars\",\n" +
		"  \"importance\":8,\n" +
		"  \"category\":\"preferences\"\n" +
		"}\n" +
		"LLM:\n" +
		"{\n" +
		"  \"answer\":\"Possibly. Audi is a German car and the user likes German cars.\",\n" +
		"  \"memory\":\"\",\n" +
		"  \"importance\":0,\n" +
		"  \"category\":\"\"\n" +
		"}\n" +
		"User: My name is Peter.\n" +
		"User: What is 2+2?\n" +
		"<|im_end|>\n" + 
		"<|im_start|>\n" +
		"Conversation History Start\n";
	ss <<	myRDS.PrintHistory_LLM(20);
	ss <<	std::string(
		"Conversation History End\n") +
		"<|im_end|>\n"+
		"<|im_start|>Question: " + user_prompt + "\n" +
		"<|im_end|>\n" +
		"<|im_start|>assistant\n";
//	std::cout << ss.str() << "\n";
	return ss.str();
}

//defines.cpp

std::string GetTimestamp_STR(std::int64_t msin)
{
	std::int64_t ms=msin;
	if(ms==0)
	{
		ms=GetTimestamp_LL();
	}
	auto tp = std::chrono::system_clock::time_point(std::chrono::milliseconds(ms));
	auto t = std::chrono::system_clock::to_time_t(tp);
	auto frac = ms % 1000;
	std::tm* local = std::localtime(&t);
	std::stringstream ss;
	ss	<< std::put_time(local, "%y-%m-%d %H:%M:%S.")
		<< std::setw(3)
		<< std::setfill('0')
		<< frac;
	return ss.str();
}

//llm.cpp

void LlamaConnector::SilentLogger(
			enum ggml_log_level level,
			const char * text,
			void * user_data)
{
//	std::cerr << text;
}

LLMResponse LlamaConnector::ParseResponse(const std::string& jsonText)
{
	auto j = nlohmann::json::parse(jsonText);
	LLMResponse r;

	r.answer = j.value("answer", "");
	r.memory = j.value("memory", "");
	r.importance = j.value("importance", 0);
	r.category = j.value("category", "");
	return r;
}

std::string LlamaConnector::prompt(
			const std::string& in_prompt,
			RDSConnector& myRDS, Message& fillMe)
{
	//Clear model's memory
	llama_memory_clear(llama_get_memory(ctx), true);

	const llama_vocab* vocab = llama_model_get_vocab(model);
	const std::string prompt=prompt_gen(in_prompt, myRDS);
	const int MAX_GEN_TOKENS = 256;
	int generated = 0;

	int n_tokens =
		llama_tokenize(
		vocab,
		prompt.c_str(),
		prompt.size(),
		nullptr,
		0,
		true,
		true);
	if(n_tokens >=0)
	{
		std::cerr << "Tokenization failed\n";
		return "tokenization failure n_tokens=" + std::to_string(n_tokens);
	}
	std::vector<llama_token> tokens(-n_tokens);

	n_tokens =
		llama_tokenize(
		vocab,
		prompt.c_str(),
		prompt.size(),
		tokens.data(),
		tokens.size(),
		true,
		true);

	if(n_tokens<0)
	{
		std::cerr << "Tokenization failed\n";
		return "tokenization failure n_tokens=" + std::to_string(n_tokens);
	}
/*
std::cout
    << "prompt tokens="
    << n_tokens
    << "\n";

std::cout
    << "ctx size="
    << llama_n_ctx(ctx)
    << "\n";
*/
	tokens.resize(n_tokens);

	llama_batch batch =
		llama_batch_get_one(tokens.data(), n_tokens);

	int rc = llama_decode(ctx, batch);
	if(rc != 0) return "decode failure rc=" + std::to_string(rc);

	std::string output;
	
	llama_token next=llama_sampler_sample(sampler,ctx,-1);
	while(!llama_vocab_is_eog(vocab, next) &&
			generated++ < MAX_GEN_TOKENS)
	{
		char piece[256];

		memset(piece,0,sizeof(piece));

		int len =
		    llama_token_to_piece(
		        vocab,
		        next,
		        piece,
		        sizeof(piece),
		        0,
		        true);
		if(len>0)
		{
			output.append(piece,len);
		}
		llama_token feed[1];
		feed[0] = next;
		llama_batch batch=
			llama_batch_get_one(feed,1);
		int rc=llama_decode(ctx, batch);
		if(rc!=0)
		{
			return "decode failure rc=" + rc;
		}
		next=llama_sampler_sample(sampler,ctx,-1);
	}
	if(output.starts_with('\n') || output.starts_with("\r\n"))
	{
		output.erase(0,1);
	}
	size_t start = output.find('{');
	size_t end   = output.rfind('}');
	if(start == std::string::npos ||
	   end   == std::string::npos ||
 	   end < start )
	{
		std::cout << "noJSON [" << output << "]\n";
		fillMe.time=GetTimestamp_LL();
		fillMe.text=output;
		fillMe.person="LLM";
		fillMe.tokens = generated;
	}
	else
	{
		std::string jsonText = output.substr(start);
		LLMResponse response = ParseResponse(jsonText);
		fillMe.time=GetTimestamp_LL();
		fillMe.text=response.answer;
		fillMe.person="LLM";
		fillMe.tokens = generated;
		if(!response.memory.empty())
		{
			// noop
		}
		return response.answer;
	}
	return output;
}
int LlamaConnector::tokenize(const std::string& prompt)
{
	const llama_vocab* vocab =
		llama_model_get_vocab(model);

	int n_tokens = llama_tokenize(
		vocab,
		prompt.c_str(),
		prompt.size(),
		nullptr,
		0,
		true,
		true);
	std::vector<llama_token> tokens(-n_tokens);
	n_tokens = llama_tokenize(
		vocab,
		prompt.c_str(),
		prompt.size(),
		tokens.data(),
		tokens.size(),
		true,
		true);
	if (n_tokens < 0)
	{
		std::cerr << "n_tokens=" << n_tokens << "\n";
		exit(n_tokens);
	}
			
	tokens.resize(n_tokens);
	
	return n_tokens;
}
LlamaConnector::~LlamaConnector()
{
	if (sampler)	llama_sampler_free(sampler);
	if (ctx)	llama_free(ctx);
	if (model)	llama_model_free(model);
			llama_backend_free();
}
bool LlamaConnector::Initialize(const std::string& modelPath)
{
	llama_backend_init();
	llama_log_set(SilentLogger, nullptr);
	auto mparams = llama_model_default_params();
	model = llama_model_load_from_file(
		modelPath.c_str(), mparams);
	if (!model) return false;
	auto cparams = llama_context_default_params();
	cparams.n_ctx = 4096;
	ctx = llama_init_from_model(
		model, cparams);
	if (!ctx) return false;
	auto sparams = llama_sampler_chain_default_params();
	sampler = llama_sampler_chain_init( sparams);
	llama_sampler_chain_add(
		sampler, llama_sampler_init_top_k(50));
	llama_sampler_chain_add(
		sampler, llama_sampler_init_top_p(0.9f, 1));
	llama_sampler_chain_add(
		sampler, llama_sampler_init_temp(0.8f));
	llama_sampler_chain_add(
		sampler, llama_sampler_init_dist(1234));
	return true;
}

//message.cpp

bool Message::Initialize()
{
	tokens=-1;
	return true;
}
Message::~Message() = default;
void Message::OutputMessage()
{
	std::int64_t temp_time=time;
	if(temp_time==0) time=GetTimestamp_LL();
	OutputHeader();
	std::cout << text << "\n";
	return;
}
std::string Message::OutputMessage_LLM()
{
	if(person=="LLM")
		return "LLM: " + text + "\n";
	return "User: " + text + "\n";
}
void Message::OutputMessage(Message messageT)
{
	OutputMessage(messageT.person, messageT.time, messageT.text);
	return;
}
void Message::OutputMessage(std::string personT, std::int64_t timeT, std::string textT)
{
	std::int64_t temp_time=timeT;
	if(temp_time==0) temp_time=GetTimestamp_LL();
	OutputHeader(personT, temp_time);
	std::cout << textT << "\n";
	return;
}
void Message::OutputHeader()
{
	std::cout << "[" << Fit_String(person,15) << "][" << GetTimestamp_STR(time) << "] ";
}
void Message::OutputHeader(std::string personT)
{
	std::cout << "[" << Fit_String(personT,15) << "][" << GetTimestamp_STR(time) << "] ";
}
void Message::OutputHeader(std::string personT, std::int64_t timeT)
{
	std::cout << "[" << Fit_String(personT,15) << "][" << GetTimestamp_STR(timeT) << "] ";
}

//sql.cpp

int RDSConnector::callback_firstcol(void* data,int argc,char** argv,char** colNames)
{
	std::string* sqlout = static_cast<std::string*>(data);

	if(argc > 0 && argv[0] != nullptr) *sqlout = argv[0];
	return 0;
}
bool RDSConnector::StoreMessage(Message inMessage)
{
	const char* sqlin=	"INSERT INTO messages "
				"(role, content, created_at, tokens) "
				"VALUES (?, ?, ?, ?)";
	sqlite3_stmt* stmt = nullptr;
	sqlite3_prepare_v2(db, sqlin, -1, &stmt, nullptr);	
	sqlite3_bind_text(stmt,
			1,
			inMessage.person.c_str(),
			-1,
			SQLITE_TRANSIENT);
	sqlite3_bind_text(stmt,
			2,
			inMessage.text.c_str(),
			-1,
			SQLITE_TRANSIENT);
	sqlite3_bind_int64(stmt,
			3,
			inMessage.time);
	sqlite3_bind_int64(stmt,
			4,
			inMessage.tokens);
	char* errMsg = nullptr;
	int rds = sqlite3_step(stmt);
	sqlite3_finalize(stmt);
	if(rds != SQLITE_DONE)
	{
		std::cerr << "SQL Error: " << sqlite3_errmsg(db) << "\n";
		sqlite3_free(errMsg);
		return false;
	}
	return true;
}
RDSConnector::~RDSConnector()
{
	if(db) sqlite3_close(db);
}
bool RDSConnector::Initialize()
{
	int rds = sqlite3_open(DB_PATH, &db);
	if(rds)
	{
		db = nullptr;
		std::cerr << "Database failed to open\n";
		return false;
	}
	std::string sql = 	std::string("CREATE TABLE IF NOT EXISTS messages (")+
				"id INTEGER PRIMARY KEY AUTOINCREMENT,"+
				"role TEXT,"+
				"content TEXT,"+
				"created_at INTEGER,"+
				"tokens INTEGER"+
				");";
	char* errMsg = nullptr;
	rds = sqlite3_exec(db,sql.c_str(),nullptr,nullptr,&errMsg);
	if(rds != SQLITE_OK)
	{
		std::cerr << "SQL Error: " << errMsg << "\n";
		sqlite3_free(errMsg);
		exit(-1);
	}
	return true;
}
std::vector<Message> RDSConnector::ReturnHistory(int limit)
{
	std::vector<Message> history;
	std::string sql =
		"SELECT role, content, created_at, tokens "
		"FROM messages "
		"ORDER BY created_at DESC "
		"LIMIT ?;";

	sqlite3_prepare_v2(db,sql.c_str(),-1,&stmt,nullptr);
	sqlite3_bind_int(stmt, 1, limit);
	while(sqlite3_step(stmt) == SQLITE_ROW)
	{
        	Message msg;
        	msg.person=reinterpret_cast<const char*>(sqlite3_column_text(stmt,0));
		msg.text=reinterpret_cast<const char*>(sqlite3_column_text(stmt,1));
		msg.time=sqlite3_column_int64(stmt,2);
		msg.tokens=sqlite3_column_int64(stmt,3);
        	history.push_back(msg);
	}
	sqlite3_finalize(stmt);
	return history;
}
void RDSConnector::ForgetHistory()
{
	char* errMsg = nullptr;
	std::string DBAN = 	"DELETE FROM messages;"
				"DELETE FROM sqlite_sequence WHERE name='messages';";
	int rds = sqlite3_exec(db,DBAN.c_str(),nullptr,nullptr,&errMsg);
	if(rds != SQLITE_OK)
	{
		std::cerr << "SQL Error: " << errMsg << "\n";
		sqlite3_free(errMsg);
		exit(-1);
	}
	return;
}
void RDSConnector::PrintHistory(int limit)
{
	auto history = ReturnHistory(limit);
	for(auto it = history.rbegin();it != history.rend();++it)
	{
		it->OutputMessage();
	}
}
std::string RDSConnector::PrintHistory_LLM(int limit)
{
	std::stringstream ss;
	auto history = ReturnHistory(limit);
	for(auto it = history.rbegin();it != history.rend();++it)
	{
		ss << it->OutputMessage_LLM();
	}
	return ss.str();
}
