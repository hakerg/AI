#pragma once
#include <vector>

class Neuron
{
public:
	float bias = 0.0f, input, output, dC_dib, dC_do, dC_dibTotal = 0.0f, df_do = 1.0f, df_di;
	std::vector<float> inputWeights, dC_dw;
	size_t inputSize;
	Neuron(const size_t & inputsCount = 0);
	inline void calculateOutput()
	{
		output = 1.0f / (1.0f + expf(-input));
	}
	inline float get_do_di()
	{
		return output * (1.0f - output);
	}
	void applyGradient(const float & factor);
	void mutate(const float & mutationRate = 0.01f);
	void combine(const Neuron & n);
	inline static float random(const float & from, const float & to);
};
