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

//memory.h

struct MemoryS
{
	std::string memory;
	std::string category;
	int importance;
	int64_t created_at;
	std::vector<float> embedding;
};

std::string printEmbedding(const std::vector<float>& vec)
{
	std::stringstream ss;
	ss << "[";
	size_t count = 20;
	for(size_t i = 0;i < std::min(count, vec.size());++i)
	{	
		if(i)
		{
			ss << ", ";
		}
		ss << vec[i];
	}
	ss << ", ...]\n";
	return ss.str();
}

void printMemoryS(MemoryS mem_in)
{
	std::cout 	<< mem_in.memory << "\n"
			<< mem_in.category << "\n"
			<< mem_in.importance << "\n"
			<< mem_in.created_at << "\n"
			<< printEmbedding(mem_in.embedding) << "\n";
}

class MemoryC
{
	private:
		LlamaConnector llmbert;
		const std::string modelPath = "models/all-minilm-l6-v2_f32.gguf";
		RDSConnector* innerRDS;
	public:
		RDSConnector getRDS()
		{
			return *innerRDS;
		}
		MemoryS mem;
		~MemoryC() = default;
		float CosineSimilarity(
			const std::vector<float>& a,
			const std::vector<float>& b)
		{
			float dot = 0.0f;
			float magA = 0.0f;
			float magB = 0.0f;

			for(size_t i = 0; i < a.size(); ++i)
			{
				dot += a[i] * b[i];
				magA += a[i] * a[i];
				magB += b[i] * b[i];
			}

			return dot /
				(std::sqrt(magA)
				* std::sqrt(magB));
		}
		void makeTables()
		{
			std::string mem_table =	std::string(
				"CREATE TABLE IF NOT EXISTS memories(")+
				"id INTEGER PRIMARY KEY AUTOINCREMENT,"+
				"memory TEXT,"+
				"category TEXT,"+
				"importance INTEGER,"+
				"created_at INTEGER"+
				");";
			char* errMsg = nullptr;
			int rc = sqlite3_exec(innerRDS->GetDB(),mem_table.c_str(),nullptr,nullptr,&errMsg);
			if(rc != SQLITE_OK)
			{
				std::cerr << "SQL Error: " << errMsg << "\n";
				sqlite3_free(errMsg);
				exit(-1);
			}
			std::string embed_table = std::string(
				"CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings\n")+
				"using vec0("+
				"embedding FLOAT[384]);";
			rc = sqlite3_exec(innerRDS->GetDB(),embed_table.c_str(),nullptr,nullptr,&errMsg);
			if(rc != SQLITE_OK)
			{
				std::cerr << "SQL Error: " << errMsg << "\n";
				sqlite3_free(errMsg);
				exit(-1);
			}
			return;
		}
		bool Initialize(RDSConnector& myRDS)
		{
			innerRDS = &myRDS;
			if(!llmbert.InitializeM(modelPath, true, 512))
			{
				std::cerr << "Decision Model failed to open\n";
				return 5;
			}
			makeTables();
			return true;
		}
		std::vector<float> Embed(const std::string& text)
		{
			return llmbert.GetEmbedding(text);
		}			
		void ForgetMemories()
		{
			sqlite3_exec(innerRDS->GetDB(),"DROP TABLE IF EXISTS memories;",nullptr,nullptr,nullptr);
			sqlite3_exec(innerRDS->GetDB(),"DROP TABLE IF EXISTS memory_embeddings;",nullptr,nullptr,nullptr);
			makeTables();
		}
		bool Store()
		{
			const char* sqlin=
				"INSERT INTO memories "
				"(memory, category, importance, created_at) "
				"VALUES (?, ?, ?, ?)";
			sqlite3_stmt* stmt = nullptr;
			sqlite3_prepare_v2(innerRDS->GetDB(), sqlin, -1, &stmt, nullptr);	
			sqlite3_bind_text(stmt,
					1,
					mem.memory.c_str(),
					-1,
					SQLITE_TRANSIENT);
			sqlite3_bind_text(stmt,
					2,
					mem.category.c_str(),
					-1,
					SQLITE_TRANSIENT);
			sqlite3_bind_int64(stmt,
					3,
					mem.importance);
			sqlite3_bind_int64(stmt,
					4,
					GetTimestamp_LL());
			char* errMsg = nullptr;
			int rc = sqlite3_step(stmt);
			sqlite3_finalize(stmt);
			if(rc != SQLITE_DONE)
			{
				std::cerr << "rc=" << rc << "\n";
				std::cerr << sqlite3_errmsg(innerRDS->GetDB()) << "\n";
				sqlite3_free(errMsg);
				return false;
			}
			sqlite3_int64 id = sqlite3_last_insert_rowid(innerRDS->GetDB());
			std::vector<float> embedding = Embed(mem.memory);
			StoreEmbed(embedding, id);
			return true;
		}
		void StoreEmbed(std::vector<float> em, sqlite3_int64 id)
		{
			const char* sqlmem =
				"INSERT INTO memory_embeddings "
				"(rowid,embedding) " 
				"VALUES (?, ?)";
			sqlite3_stmt* stmt = nullptr;
			sqlite3_prepare_v2(innerRDS->GetDB(), sqlmem, -1, &stmt, nullptr);	
			sqlite3_bind_int64(stmt,
					1,
					id);
			sqlite3_bind_blob(stmt,
					2,
					em.data(),
					em.size() * sizeof(float),
					SQLITE_TRANSIENT);
			char* errMsg = nullptr;
			int rc = sqlite3_step(stmt);
			sqlite3_finalize(stmt);
			if(rc != SQLITE_DONE)
			{
				std::cerr << "SQL Error: " << sqlite3_errmsg(innerRDS->GetDB()) << "\n";
				sqlite3_free(errMsg);
				exit(-1);
			}
			return;
		}
		std::vector<MemoryS> Retrieve(std::string text_in, int max_mem)
		{
			std::stringstream ss;
			std::string memory_list;
//			ss << "Message History Start\n";
//			ss << innerRDS->PrintHistory_LLM(20);
//			ss << "Message History End\n";
//			ss << "Most recent user prompt Start\n";
			ss << text_in << "\n";
//			ss << "Most recent user prompt End\n";
			auto queryVec = Embed(ss.str());

			struct ScoredMemory
			{
				float score;
				MemoryS mem;
			};

			std::vector<ScoredMemory> scored;
			auto memories = RetrieveAll();
			for(auto& mem : memories)
			{
				float score =
					CosineSimilarity(queryVec, mem.embedding);
				ScoredMemory sm;
				sm.score = score;
				sm.mem = mem;
				scored.push_back(sm);
			}
			std::sort(	scored.begin(),
					scored.end(),
					[](auto& a, auto& b)
			{
				return a.score > b.score;
			});
			std::vector<MemoryS> result;
			for(size_t i = 0; i<scored.size() &&
					i< max_mem; ++i)
			{
				result.push_back(scored[i].mem);
//				printMemoryS(scored[i].mem);
			}

			return result;
		}
		std::vector<MemoryS> RetrieveAll()
		{
			std::vector<MemoryS> out;
			sqlite3_stmt* stmt = nullptr;

			sqlite3_prepare_v2(
				innerRDS->GetDB(),
					"SELECT m.id, m.memory, m.category, m.importance, m.created_at, e.embedding "
					"FROM memories m "
					"JOIN memory_embeddings e "
					"ON e.rowid = m.id;",
				-1,
				&stmt,
				nullptr);

			while(sqlite3_step(stmt)== SQLITE_ROW)
			{
				MemoryS mem;

				mem.memory =
					reinterpret_cast<const char*>(
					sqlite3_column_text(stmt,1));
				mem.category =
					reinterpret_cast<const char*>(
					sqlite3_column_text(stmt,2));
				mem.importance = sqlite3_column_int(stmt,3);
				mem.created_at = sqlite3_column_int64(stmt,4);
				const void* blob = sqlite3_column_blob(stmt,5);
				int bytes = sqlite3_column_bytes(stmt,5);
				mem.embedding.resize(bytes/sizeof(float));
				memcpy(mem.embedding.data(),blob,bytes);
				out.push_back(mem);
			}
			sqlite3_finalize(stmt);

			return out;
		}
		std::vector<MemoryS> RetrieveAll(int max)
		{
			std::vector<MemoryS> out;
			sqlite3_stmt* stmt = nullptr;

			sqlite3_prepare_v2(
				innerRDS->GetDB(),
					"SELECT m.id, m.memory, m.category, m.importance, m.created_at, e.embedding "
					"FROM memories m "
					"JOIN memory_embeddings e "
					"ON e.rowid = m.id;",
				-1,
				&stmt,
				nullptr);
			int i = 1;
			while(sqlite3_step(stmt)== SQLITE_ROW && i++ < max)
			{
				MemoryS mem;

				mem.memory =
					reinterpret_cast<const char*>(
					sqlite3_column_text(stmt,1));
				mem.category =
					reinterpret_cast<const char*>(
					sqlite3_column_text(stmt,2));
				mem.importance = sqlite3_column_int(stmt,3);
				mem.created_at = sqlite3_column_int64(stmt,4);
				const void* blob = sqlite3_column_blob(stmt,5);
				int bytes = sqlite3_column_bytes(stmt,5);
				mem.embedding.resize(bytes/sizeof(float));
				memcpy(mem.embedding.data(),blob,bytes);
				out.push_back(mem);
			}
			sqlite3_finalize(stmt);

			return out;
		}
};

struct LLMResponse
{
	std::string answer;
	MemoryS mems;
};

//prompt_gen.cpp

std::string prompt_gen(const std::string user_prompt, RDSConnector& myRDS)
{
	std::stringstream ss;
	MemoryC memc;
	memc.Initialize(myRDS);

	ss << R"(<|im_start|>system
You MUST format your response in valid JSON.
For every user message:
1. Answer the user.
2. Determine whether the message contains a memory.
A memory is:
- a preference
- a personal fact
- a goal
- a relationship
- a recurring interest
If a memory exists:
- you MUST store your answer in the answer field.
- you MUST store the memory in the memory field.
- you MUST store importance in the imporance field. importance MUST be a number between 1-10.
- you MUST store category in the category field.
If this should not be a memory:
- you MUST store your answer in the answer field.
- you MUST store "" in the memory fied.
- you MUST store 0 in the importance field.
- you MUST store "" in the category field.
Remember to format your entire response as JSON. 
Here are some examples.
User: What is 2+2?
LLM:
{
	"answer":"2+2=4",
	"memory":"",
	"importance":0,
	"category":""
}
User: My name is Peter.
LLM:
{
	"answer":"Nice to meet you.",
	"memory":"User's name is Peter",
	"importance":10,
	"category":"identity"
}
User: I like German cars.
LLM:
{
	"answer":"Interesting.",
	"memory":"User likes German cars",
	"importance":8,
	"category":"preferences"
}
User: Do I like Audi?
LLM:
{
	"answer":"Possibly. Audi is a German car and the user likes German cars.",
	"memory":"",
	"importance":0,
	"category":""
}
<|im_end|>
)";
	ss <<	"<|im_start|>\n"
		<< "Conversation History Start\n";
	ss <<	myRDS.PrintHistory_LLM(20);
	ss <<	"Conversation History End\n"
		<< "<|im_end|>\n";

	ss <<	"Relevant Memories Start\n";
		auto memorieZ = memc.Retrieve(user_prompt, 10);
		for(auto& mem : memorieZ)
		{
			ss << mem.memory << "\n";
		}
	ss <<	"Relevant Memories End\n"
		<< "<|im_end|>\n";
	ss 	<< "<|im_start|>user\n" 
		<< user_prompt + "\n" 
		<< "<|im_end|>\n";
	ss	<< "<|im_start|>assistant\n";
	std::cout << ss.str() << "\n"; 
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

std::vector<float>
LlamaConnector::GetEmbedding(
	const std::string& text)
{
	std::vector<float> output;

	const llama_vocab* vocab =
		llama_model_get_vocab(
			model);

	int n_tokens =
		llama_tokenize(
			vocab,
			text.c_str(),
			text.size(),
			nullptr,
			0,
			true,
			true);

	if(n_tokens >= 0)
	{
		return {};
	}

	n_tokens = -n_tokens;

	std::vector<llama_token>
		tokens(n_tokens);

	llama_tokenize(
		vocab,
		text.c_str(),
		text.size(),
		tokens.data(),
		tokens.size(),
		true,
		true);

	llama_memory_clear(
		llama_get_memory(ctx),
		true);

	llama_batch batch =
		llama_batch_get_one(
			tokens.data(),
			tokens.size());

	if(llama_decode(ctx,batch) != 0)
	{
		return {};
	}

	float* emb = llama_get_embeddings_seq(ctx,0);
/*	std::cout
	<< "emb ptr="
	<< emb
	<< "\n";
	std::cout << "pooling=" << llama_pooling_type(ctx) << "\n";
*/
	if(!emb)
	{
		return {};
	}

	int dim =
		llama_model_n_embd(
			model);

	output.resize(dim);

	for(int i=0;i<dim;i++)
	{
		output[i] = emb[i];
	}

	return output;
}

LLMResponse LlamaConnector::ParseResponse(const std::string jsonText)
{
	auto j = nlohmann::json::parse(jsonText);
	LLMResponse r;

	r.answer = j.value("answer", "");
	r.mems.memory = j.value("memory", "");
	r.mems.importance = j.value("importance", 0);
	r.mems.category = j.value("category", "");
	r.mems.created_at = GetTimestamp_LL();
	return r;
}

std::string LlamaConnector::prompt(
			const std::string& in_prompt,
			RDSConnector& myRDS, Message& fillMe)
{
	//Clear model's memory
//	llama_memory_clear(llama_get_memory(ctx), true);
	MemoryC mem_now;
	mem_now.Initialize(myRDS);


	const llama_vocab* vocab = llama_model_get_vocab(model);
	const std::string prompt=prompt_gen(in_prompt, myRDS);
	const int MAX_GEN_TOKENS = 366;
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
	size_t end	= output.rfind('}');
	if(start == std::string::npos ||
		end	== std::string::npos ||
 		end < start )
	{
		std::cout << "noJSON [" << output << "]\n";
		fillMe.time=GetTimestamp_LL();
		if(output.empty()) fillMe.text="[EMPTY_RESPONSE]";
		else fillMe.text=output;
		fillMe.person="LLM";
		fillMe.tokens = generated;
	}
	else
	{
		std::string jsonText = output.substr(start);
		LLMResponse response = ParseResponse(jsonText);
		mem_now.mem=response.mems;	
		mem_now.Store();
		fillMe.time=GetTimestamp_LL();
		fillMe.text=response.answer;
		fillMe.person="LLM";
		fillMe.tokens = generated;
		if(!response.mems.memory.empty())
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
bool LlamaConnector::InitializeM(const std::string& modelPath, bool em, int nctx)
{
	llama_backend_init();
	llama_log_set(SilentLogger, nullptr);
	auto mparams = llama_model_default_params();
	model = llama_model_load_from_file(
		modelPath.c_str(), mparams);
	if (!model) return false;
	auto cparams = llama_context_default_params();
	cparams.embeddings = em;
	cparams.n_ctx = 512;
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
sqlite3* RDSConnector::GetDB()
{
	return db;
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
