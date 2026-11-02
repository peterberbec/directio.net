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
	LlamaConnector llm;

//	if(!llm.Initialize("models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
	if(!llm.Initialize("models/Qwen3.8-0.6B.Q4_K_M.gguf"))
	{
		std::cerr << "Model failed to open\n";
		return 5;
	}

	RDSConnector myRDS;
	if(!myRDS.Initialize())
	{
		std::cerr << "RDS failed to open\n";
		return 5;
	}

	myMessage.person = "System";
	myMessage.text = "Enter prompt. \"!quit\" to quit.";
	myMessage.time = GetTimestamp_LL();
	myMessage.OutputMessage();
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
		else if(myMessage.text.starts_with("!forget"))
		{
			myRDS.ForgetHistory();
			myMessage.OutputMessage("SYSTEM",GetTimestamp_LL(),
					"Conversation history cleared.");
			continue;
		}
		else if(myMessage.text.starts_with("!history"))
		{
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
		// Let's get the reply
		llm.prompt(myMessage.text,myRDS,llmMessage);
		// store the messages
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
