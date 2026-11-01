#pragma once

#include <iostream>

std::string prompt_gen(const std::string user_prompt)
{
	return "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n" + user_prompt + "\n<|im_end|><|im_start|>assistant";
}
