#include <iostream>
#include "llama.h"
#include "header.h"
#include "include.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>


int main()
{
	std::string prompt = "";
	std::string user = "Peter Berbec";
	Message myMessage, llmMessage;
	bool exittime = false;
	std::int64_t timestamp;
	int tokens;
	LlamaConnector llm, decide;

	if(!llm.Initialize("models/Qwen2.5-0.5b-instruct-q4_k_m.gguf"))
	{
		std::cerr << "Main Model failed to open\n";
		return 5;
	}

	RDSConnector myRDS;
	if(!myRDS.Initialize())
	{
		std::cerr << "RDS failed to open\n";
		return 9;
	}


	sqlite3_enable_load_extension(myRDS.GetDB(), 1);
	char* err = nullptr;
	sqlite3_load_extension(myRDS.GetDB(), "C:\\msys64\\home\\peter\\AI_coding\\directio.net\\lib\\vec0.dll", nullptr, &err);
	if(err)
	{
		std::cerr << err << "\n";
	}
	MemoryC memc;
	memc.Initialize(myRDS);
	myMessage.OutputMessage("SYSTEM",GetTimestamp_LL(),
				"Enter prompt. \"!help\" for help.");
	while(!exittime)
	{
		myMessage.time = GetTimestamp_LL();
		myMessage.OutputHeader(user);
		myMessage.person = user;
		std::getline(std::cin, myMessage.text); std::cout << "\033[A\033[K";
		myMessage.time = GetTimestamp_LL();
		myMessage.OutputMessage();

		if(myMessage.text.starts_with("!quit"))
		{
			exittime = true;
			continue;
		}
		else if(myMessage.text.starts_with("!help"))
		{
			myMessage.OutputMessage("!help",GetTimestamp_LL(),
				"!quit    - to exit");
			myMessage.OutputMessage("!help",GetTimestamp_LL(),
				"!memory  - to show all memories");
			myMessage.OutputMessage("!help",GetTimestamp_LL(),
				"!history - to show all history");
			myMessage.OutputMessage("!help",GetTimestamp_LL(),
				"!forget  - to clear memory/conversation history");
			myMessage.OutputMessage("!help",GetTimestamp_LL(),
				"!model   - to switch models (todo)");
			continue;
		}
		else if(myMessage.text.starts_with("!forget"))
		{
			myRDS.ForgetHistory();
			memc.ForgetMemories();
			myMessage.OutputMessage("!forget",GetTimestamp_LL(),
					"Conversation history cleared.");
			continue;
		}
		else if(myMessage.text.starts_with("!mem"))
		{
			auto memories = memc.RetrieveAll();
			for(auto& mem : memories)
			{
				std::stringstream ms;
				ms << "[" << std::setw(3) << std::setfill('0')
				   << std::to_string(mem.importance) << "] " 
				   << mem.memory;
				myMessage.OutputMessage("!memory",GetTimestamp_LL(),
					ms.str());
			}
			continue;
		}
		else if(myMessage.text.starts_with("!model"))
		{
			myMessage.OutputMessage("!model",GetTimestamp_LL(),
					"TODO");
			continue;
		}
		else if(myMessage.text.starts_with("!history"))
		{
			myMessage.OutputMessage("!history",GetTimestamp_LL(),
					"");
			int count = 20;
			std::stringstream ss(myMessage.text);
			std::string cmd;
			ss >> cmd;
			if(ss >> count)
			{
				myRDS.PrintHistory(count);
			}
			else
			{
				myRDS.PrintHistory();
			}
			continue;
		}
		myMessage.tokens = llm.tokenize(myMessage.text);
		llm.prompt(myMessage.text,myRDS,llmMessage);
		if(!myRDS.StoreMessage(myMessage))
		{
			return -1;
		}
		if(!myRDS.StoreMessage(llmMessage))
		{
			return -1;
		}
		llmMessage.OutputMessage();
	}
	myMessage.OutputMessage("SYSTEM",GetTimestamp_LL(),"Goodbye");

	return 0;
}
