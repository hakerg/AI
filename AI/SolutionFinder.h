#pragma once
#include "NeuralNetwork.h"

class SolutionFinder
{
	NeuralNetwork * fitnessNetwork;
	float * networkInput;

public:
	const size_t inputSize, outputSize, networkInputSize;

	SolutionFinder(const size_t & inputCount, const size_t & outputCount, std::vector<size_t> fitnessHiddenNetworkLayers);
	~SolutionFinder();
	void copyInput(float * input);
	void getBestOutput(float * output, const int & attempts = 10, const float & factor = 0.5f);
	void giveFitness(float fitness);
	void giveFitness(float * input, float * output, const float & fitness);
	void adjustFitnessNetwork(const float & factor = 1.0f);
};

