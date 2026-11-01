#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "message.h"
#include "defines.h"
#include <sqlite3.h>
#define DB_PATH "data/directio.net.db"

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
		int ReturnHistory(Message inMsg)
		{
			int n_message=1;
			std::string sqlin = 	std::string("SELECT role, content, created_at ")+
						"FROM messages " +
						"ORDER BY created_at DESC " +
						"LIMIT " + std::to_string(n_message) + ";";
			std::string user, text;
			long long time;
			sqlite3_prepare_v2(db, sqlin.c_str(), -1, &stmt, nullptr);
			if(sqlite3_step(stmt) == SQLITE_ROW)
			{
				inMsg.person = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
				inMsg.text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
				inMsg.time = sqlite3_column_int64(stmt, 2);
				n_message++;
			}
			sqlite3_finalize(stmt);
			return n_message;
		}
		int PrintHistory()
		{
			Message internalMsg;
			ReturnHistory(internalMsg);
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
