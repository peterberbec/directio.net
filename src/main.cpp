#include <iostream>
#include "llama.h"
#include "llm.h"
#include "sql.h"
#include "defines.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

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



int main()
{
	std::string prompt = "", sqlin = "", sqlout = "";
	std::string user = "Peter Berbec";
	Message myMessage;
	bool exittime = false;
	long long timestamp;
	int tokens;
	LlamaConnector llm;

	if(!llm.Initialize("models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
	{
		std::cerr << "Model failed to open\n";
		return 5;
	}


	myMessage.person = "System";
	myMessage.text = "Enter prompt. \"!quit\" to quit.";
	myMessage.time = GetTimestamp_LL();
	myMessage.OutputMessage();
	while(!exittime)
	{
		RDSConnector myRDS;
		myMessage.OutputHeader(user);
		myMessage.person = user;
		std::getline(std::cin, myMessage.text);
		myMessage.time = GetTimestamp_LL();

		if(myMessage.text=="!quit")
		{
			exittime = true;
			continue;
		}
		else if(myMessage.text=="!history")
		{
			myRDS.PrintHistory();
			continue;
		}
		// store the message
		if(!myRDS.StoreMessage(myMessage))
		{
			return -1;
		}
		// Let's get the reply
		myMessage.person = "LLM";
		myMessage.text = llm.prompt(prompt);
		myMessage.time = GetTimestamp_LL();
		if(!myRDS.StoreMessage(myMessage))
		{
			return -1;
		}
		myMessage.OutputMessage();
	}
	myMessage.OutputMessage("SYSTEM",GetTimestamp_LL(),"Goodbye");

	return 0;
}
