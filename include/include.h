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
#define GetTimestamp_LL() (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count())

//defines.h

std::string GetTimestamp_STR()
{
	auto now = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&t),"%y-%m-%d %H:%M:%S");
	return ss.str();
}

std::string GetTimestamp_STR(long long tsin)
{
	if(tsin==0) return GetTimestamp_STR();
	std::time_t t = static_cast<std::time_t>(tsin);
	std::tm* local = std::localtime(&t);
	std::stringstream ss;
	ss << std::put_time(local, "%y-%m-%d %H:%M:%s");
	return ss.str();
}

//llm.h

class LlamaConnector
{
	private:
		llama_model* model = nullptr;
		llama_context* ctx = nullptr;
		llama_sampler* sampler = nullptr;
		static void SilentLogger(
			enum ggml_log_level level,
			const char * text,
			void * user_data)
		{
		}
	public:
		std::string prompt(
			const std::string& in_prompt)
		{
			const llama_vocab* vocab = llama_model_get_vocab(model);
			const std::string prompt=prompt_gen(in_prompt);
 
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

			tokens.resize(n_tokens);

			llama_batch batch =
				llama_batch_get_one(tokens.data(), n_tokens);

			int rc = llama_decode(ctx, batch);
			if(rc != 0) return "decode failure rc=" + std::to_string(rc);

			std::string output;
			
			llama_token next=llama_sampler_sample(sampler,ctx,-1);
			while(!llama_vocab_is_eog(vocab, next))
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
					return "decode failure rc=" + std::to_string(rc);
				}
				next=llama_sampler_sample(sampler,ctx,-1);
			}
			if(output.starts_with('\n') || output.starts_with("\r\n"))
			{
				output.erase(0,1);
			}

			return output;
		}
		int tokenize(const std::string& prompt)
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
		~LlamaConnector()
		{
			if (sampler)	llama_sampler_free(sampler);
			if (ctx)	llama_free(ctx);
			if (model)	llama_model_free(model);
					llama_backend_free();
		}
		bool Initialize(const std::string& modelPath)
		{
			llama_backend_init();
			llama_log_set(SilentLogger, nullptr);
			auto mparams = llama_model_default_params();
			model = llama_model_load_from_file(
				modelPath.c_str(), mparams);
			if (!model) return false;
			auto cparams = llama_context_default_params();
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
				sampler, llama_sampler_init_temp(0.2f));
			llama_sampler_chain_add(
				sampler, llama_sampler_init_dist(1234));

			return true;
		}
};

//message.h

class Message
{
	public:	std::string person = "";
		std::string text = "";
		long long time = 0;
		bool Initialize()
		{
			return true;
		}
		~Message() = default;
		void OutputMessage()
		{
			long long temp_time=time;
			if(temp_time==0) time=GetTimestamp_LL();
			OutputHeader();
			std::cout << text << "\n";
			return;
		}
		void OutputMessage(Message messageT)
		{
			OutputMessage(messageT.person, messageT.time, messageT.text);
			return;
		}
		void OutputMessage(std::string personT, long long timeT, std::string textT)
		{
			long long temp_time=timeT;
			if(temp_time==0) temp_time=GetTimestamp_LL();
			OutputHeader(personT, temp_time);
			std::cout << textT << "\n";
			return;
		}
		void OutputHeader()
		{
			std::cout << "[" << Fit_String(person,15) << "][" << GetTimestamp_STR(time) << "] ";
		}
		void OutputHeader(std::string personT)
		{
			std::cout << "[" << Fit_String(personT,15) << "][" << GetTimestamp_STR(time) << "] ";
		}
		void OutputHeader(std::string personT, long long timeT)
		{
			std::cout << "[" << Fit_String(personT,15) << "][" << GetTimestamp_STR(timeT) << "] ";
		}
};

//prompt_gen.h

std::string prompt_gen(const std::string user_prompt)
{
	return "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n" + user_prompt + "\n<|im_end|><|im_start|>assistant";
}

//sql.h


class RDSConnector
{
	private:
		sqlite3* db;
		sqlite3_stmt* stmt;
	public:
		bool StoreMessage(Message inMessage)
		{
			std::string sqlin=	std::string("INSERT INTO messages (role, content, created_at) ") +
						"VALUES ('" + inMessage.person + "','" + inMessage.text + "', " + std::to_string(inMessage.time) + ");"; 
			char** errMsg;
			int rds = sqlite3_exec(db,sqlin.c_str(),nullptr,nullptr,errMsg);
			if(rds != SQLITE_OK)
			{
				std::cerr << "SQL Error: " << errMsg << "\n";
				sqlite3_free(errMsg);
				return false;
			}
			return true;
		}
		~RDSConnector()
		{
			char** errMsg;
			sqlite3_free(errMsg);
		}
		bool Initialize()
		{
			return true;
			int rds = sqlite3_open(DB_PATH, &db);
			if(rds)
			{
				std::cerr << "Database failed to open\n";
				return false;
			}
			return true;
			std::string sql = 	std::string("CREATE TABLE IF NOT EXISTS messages (")+
						"id INTEGER PRIMARY KEY AUTOINCREMENT,"+
						"role TEXT,"+
						"content TEXT,"+
						"created_at INTEGER"+
						");";
			char* errMsg = nullptr;
			rds = sqlite3_exec(db,sql.c_str(),nullptr,nullptr,&errMsg);
			if(rds != SQLITE_OK)
			{
				std::cerr << "SQL Error: " << errMsg << "\n";
				sqlite3_free(errMsg);
				exit(-1);
			}
		}
		int ReturnMessage(Message inMsg)
		{
			std::string sqlin = 	std::string("SELECT role, content, created_at ")+
						"FROM messages " +
						"ORDER BY created_at DESC " +
						"LIMIT 1;";
			std::string user, text;
			long long time;
			sqlite3_prepare_v2(db, sqlin.c_str(), -1, &stmt, nullptr);
			if(sqlite3_step(stmt) == SQLITE_ROW)
			{
				inMsg.person = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
				inMsg.text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
				inMsg.time = sqlite3_column_int64(stmt, 2);
			}
			sqlite3_finalize(stmt);
			return 1;
		}
		int PrintHistory()
		{
			Message internalMsg;
			ReturnMessage(internalMsg);
			internalMsg.OutputMessage();
			return 1;
		}
/*		int ReturnHistory(Message inMSG[], const int n_messages)
		{
			const char* sqlin = 	"SELECT role, content, created_at "
						"FROM messages "
						"ORDER BY created_at DESC "
						"LIMIT " n_messages ";";
			std::string user, text;
			long long time;
			sqlite3_prepare_v2(db, sqlin.c_str(), -1, &stmt, nullptr);
			int pos = 0;
			if(sqlite3_step(stmt) == SQLITE_ROW)
			{
				inMsg[pos].user = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
				inMsg[pos].text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
				inMsg[pos].time = sqlite3_column_int64(stmt, 2);
				pos++;
				if(pos>=n_messages) break;
			}
			sqlite3_finalize(stmt);
		} 
*/
};
