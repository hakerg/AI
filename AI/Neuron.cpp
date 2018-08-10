#include "Neuron.h"
#include <stdlib.h>
#include <string>



Neuron::Neuron(const size_t & inputsCount) : inputWeights(inputsCount), dC_dw(inputsCount), inputSize(inputsCount)
{
	float range = 1.0f / inputsCount;
	for (auto& w : inputWeights) w = random(-range, range);
}

void Neuron::applyGradient(const float & factor)
{
	bias -= dC_dibTotal * factor;
	dC_dibTotal = 0.0f;
	for (size_t w = 0; w < inputSize; w++)
	{
		float& wv = dC_dw[w];
		inputWeights[w] -= wv * factor;
		wv = 0.0f;
	}
}

void Neuron::mutate(const float & mutationRate)
{
	bias += random(-mutationRate, mutationRate);
	for (float& w : inputWeights)
	{
		w += random(-mutationRate, mutationRate);
	}
}

void Neuron::combine(const Neuron & n)
{
	if (inputSize != n.inputSize) throw std::string("Invalid argument");
	bias = random(bias, n.bias);
	for (size_t w = 0; w < inputSize; w++)
	{
		inputWeights[w] = random(inputWeights[w], n.inputWeights[w]);
	}
}

float Neuron::random(const float & from, const float & to)
{
	return rand() / float(RAND_MAX) * (to - from) + from;
}
