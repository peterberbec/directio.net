#include <iostream>
#include "llama.h"
#include <sqlite3.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

class LlamaConnector
{
	private:
		llama_model* model;
		llama_context* ctx;
		llama_sampler* sampler;
		static void SilentLogger(
			enum ggml_log_level level,
			const char * text,
			void * user_data)
		{
		}
	public:
		std::string Prompt(
			const std::string& prompt);
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
				sampler, llama_sampler_init_temp(0.8f));
			llama_sampler_chain_add(
				sampler, llama_sampler_init_dist(1234));

			return true;
		}

};


int callback_firstcol(
		void* data,
		int argc,
		char** argv,
		char** colNames)
{
	std::string* sqlout = static_cast<std::string*>(data);

	if(argc > 0 && argv[0] != nullptr)
	{
		*sqlout = argv[0];
	}

	return 0;
}

std::string GetTimestamp_STR()
{
		auto now = std::chrono::system_clock::now();

		std::time_t t =
	std::chrono::system_clock::to_time_t(now);

		std::stringstream ss;

		ss << std::put_time(
	std::localtime(&t),
	"%Y-%m-%d %H:%M:%S");

		return ss.str();
}

std::string LL_timestamp_STR(long long tsin)
{
	std::time_t t = static_cast<std::time_t>(tsin);
	std::tm* local = std::localtime(&t);
	std::stringstream ss;
	ss << std::put_time(local, "%y-%m-%d %H:%M%s");
	return ss.str();
}

long long GetTimestamp_LL()
{
	return std::chrono::duration_cast<
		std::chrono::seconds>(
		std::chrono::system_clock::now()
			.time_since_epoch()).count();
}

int main()
{
	std::string prompt = "", sqlin = "", sqlout = "";
	std::string user = "Peter Berbec";
	bool exittime = false;
	sqlite3* db;
	sqlite3_stmt* stmt;
	long long timestamp;

	LlamaConnector llm;
	llm.Initialize("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
	return 0;
	llama_backend_init();
	llama_log_set(
			[](enum ggml_log_level, const char *, void*) {}, nullptr
	);
	llama_model* model = nullptr;
	llama_model_params model_params = llama_model_default_params();

	model = llama_model_load_from_file("models/qwen2.5-0.5b-instruct-q4_k_m.gguf", model_params);

	if(!model)
	{
		std::cerr << "Model failed to open\n";
		return 5;
	}
	std::cout << "model loaded\n";

	llama_context_params ctx_params =
		llama_context_default_params();

	llama_context* ctx = llama_init_from_model(model, ctx_params);
	if(!ctx)
	{
		std::cerr << "Failed to create context\n";
		return 1;
	}
	std::cout << "Context Created\n";

	int rds = sqlite3_open("data/directio.net.db", &db);

	if(rds)
	{
		std::cerr << "Database failed to open\n";
		return 1;
	}

	const char* sql = 
		"CREATE TABLE IF NOT EXISTS messages ("
		"id INTEGER PRIMARY KEY AUTOINCREMENT,"
		"role TEXT,"
		"content TEXT,"
		"created_at INTEGER"
		");";

	char* errMsg = nullptr;

	rds = sqlite3_exec(
			db,
			sql,
			nullptr,
			nullptr,
			&errMsg);

	if(rds != SQLITE_OK)
	{
		std::cerr << "SQL Error: " << errMsg << "\n";
		sqlite3_free(errMsg);
	}

	std::cout << "Database Initialized\n";

	while(!exittime)
	{
		std::cout << "Enter prompt. \"quit\" to quit.\n";
		std::getline(std::cin, prompt);
		timestamp=GetTimestamp_LL();

		sqlin=
			"INSERT INTO messages (role, content, created_at) "
			"VALUES ('" + user + "','" + prompt + "', " + std::to_string(timestamp) + ");"; 

		if(prompt == "quit")
		{
			exittime = true;
			continue;
		}

		rds = sqlite3_exec(
				db,
				sqlin.c_str(),
				nullptr,
				nullptr,
				&errMsg);

		if(rds != SQLITE_OK)
		{
			std::cerr << "SQL Error: " << errMsg << "\n";
			sqlite3_free(errMsg);
		}

		sqlin = "SELECT role, content, created_at "
		"FROM messages "
		"ORDER BY created_at DESC "
		"LIMIT 1;";

		sqlite3_prepare_v2(db, sqlin.c_str(), -1, &stmt, nullptr);
		if(sqlite3_step(stmt) == SQLITE_ROW)
		{
			std::cout << "You entered:\n";
			std::string role = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
			std::string content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
			long long timestamp = sqlite3_column_int64(stmt, 2);
			std::cout << "[" << role << "][" << LL_timestamp_STR(timestamp) << "] " << content << "\n";
		}

		sqlite3_finalize(stmt);

	}
	std::cout << "Goodbye\n";

	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
	return 0;
}
