#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>

#define Fit_String(s,w) (((s).size()>=(w))?(s.substr(0,(w))):((s)+std::string((w)-(s).size(),' ')))
#define GetTimestamp_LL() (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count())

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
