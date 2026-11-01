#!/bin/bash
#g++ -g -O0 src/main.cpp src/llm.cpp  -Iinclude   -Llib   -lllama   -lggml   -lggml-base   -lggml-cpu   -lsqlite3   -o directio_net.exe
cmake -B build
cmake --build build
