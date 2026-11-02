#!/bin/bash
set -o pipefail
cd /home/peter/AI_coding/directio.net
vi ./src/main.cpp ./include/include.h ./include/header.h
cmake -B build 2>&1 | tee _buildlog.txt
good1=$?
cmake --build build 2>&1 | tee -a _buildlog.txt
good2=$?
if [ $good1 -eq 0 ] && [ $good2 -eq 0 ]
then
	script -c "./bin/directio.net.exe" _runlog.txt
fi
cp /home/peter/AI_coding/directio.net/CMakeLists.txt /c/llm/
cp /home/peter/AI_coding/directio.net/src/* /c/llm/
cp /home/peter/AI_coding/directio.net/include/* /c/llm/
cp /home/peter/AI_coding/directio.net/_buildlog.txt /c/llm/
cp /home/peter/AI_coding/directio.net/_runlog.txt /c/llm/
