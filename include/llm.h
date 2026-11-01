#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "llama.h"
#include "prompt_gen.h"

class LlamaConnector
{
	private:
		llama_model* model = nullptr;
		llama_context* ctx = nullptr;
		llama_sampler* sampler = nullptr;
		static void SilentLogger(
			enum ggml_log_level level,
			const char * text,
			void * user_data)
		{
		}
	public:
		std::string prompt(
			const std::string& in_prompt)
		{
			const llama_vocab* vocab = llama_model_get_vocab(model);
			const std::string prompt=prompt_gen(in_prompt);
 
			int n_tokens =
				llama_tokenize(
				vocab,
				prompt.c_str(),
				prompt.size(),
				nullptr,
				0,
				true,
				true);
			// tokenize
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
/*			else
			{
				std::cout << "Prompt tokens: "
					<< n_tokens << "\n";
			}
*/

			tokens.resize(n_tokens);

			// build batch
			llama_batch batch =
				llama_batch_get_one(tokens.data(), n_tokens);

			int rc = llama_decode(ctx, batch);
			if(rc != 0) return "decode failure rc=" + std::to_string(rc);

			std::string output;
			
/*			for(int i=0; i<50; i++)
			{
				llama_token next=
					llama_sampler_sample(sampler,ctx,-1);
				if(llama_vocab_is_eog(vocab, next))
				{
					break;
				}
*/				
			llama_token next=llama_sampler_sample(sampler,ctx,-1);
			while(!llama_vocab_is_eog(vocab, next))
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
					return "decode failure rc=" + std::to_string(rc);
				}
				next=llama_sampler_sample(sampler,ctx,-1);
			}
			if(output.starts_with('\n') || output.starts_with("\r\n"))
			{
				output.erase(0,1);
			}

			return output;
		}
		int tokenize(const std::string& prompt)
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
				sampler, llama_sampler_init_temp(0.2f));
			llama_sampler_chain_add(
				sampler, llama_sampler_init_dist(1234));

			return true;
		}
};
