#pragma once

#include <iostream>
#include "defines.h"

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
