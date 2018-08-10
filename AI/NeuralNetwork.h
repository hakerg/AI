#pragma once
#include "Neuron.h"

class NeuralNetwork
{
	std::vector<std::vector<Neuron>> neurons;
	bool calculated = false;
	int gradients = 0;

	void calculate();

public:
	const size_t inputSize, outputSize;

	NeuralNetwork(const std::vector<size_t> & layers);
	~NeuralNetwork();
	void calculateGradient(const float * expectedOutput);
	float getError(const float * expectedOutput);
	float getLastError();
	void adjustNetwork(float factor = 1.0f);
	void copyInput(const float * source);
	void getOutput(float * dest);
	void getOutput(float * input, float * output);
	void mutate(const float & mutationRate = 0.01f);
	void combine(const NeuralNetwork & nn);
	const std::vector<std::vector<Neuron>> & getNeurons();
	NeuralNetwork & operator=(const NeuralNetwork & nn);
	void findBestInput(const size_t & firstInputIndex, const size_t & inputCount, const int & attempts, const float & factor);
};

