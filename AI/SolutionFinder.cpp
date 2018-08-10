#include "SolutionFinder.h"



SolutionFinder::SolutionFinder(const size_t & inputCount, const size_t & outputCount, std::vector<size_t> fitnessHiddenNetworkLayers) : inputSize(inputCount), outputSize(outputCount), networkInputSize(inputCount + outputCount)
{
	fitnessHiddenNetworkLayers.insert(fitnessHiddenNetworkLayers.begin(), networkInputSize);
	fitnessHiddenNetworkLayers.push_back(1);
	fitnessNetwork = new NeuralNetwork(fitnessHiddenNetworkLayers);
	networkInput = new float[networkInputSize];
}


SolutionFinder::~SolutionFinder()
{
	delete fitnessNetwork;
	delete[] networkInput;
}

void SolutionFinder::copyInput(float * input)
{
	memcpy(networkInput, input, sizeof(float) * inputSize);
	fitnessNetwork->copyInput(networkInput);
}

void SolutionFinder::getBestOutput(float * output, const int & attempts, const float & factor)
{
	fitnessNetwork->findBestInput(inputSize, outputSize, attempts, factor);
	auto& in = fitnessNetwork->getNeurons().front();
	for (auto it = in.begin() + inputSize; it != in.end(); it++)
	{
		*output = it->output;
		output++;
	}
}

void SolutionFinder::giveFitness(float fitness)
{
	fitnessNetwork->calculateGradient(&fitness);
}

void SolutionFinder::giveFitness(float * input, float * output, const float & fitness)
{
	memcpy(networkInput, input, sizeof(float) * inputSize);
	memcpy(networkInput + inputSize, output, sizeof(float) * outputSize);
	fitnessNetwork->copyInput(networkInput);
	giveFitness(fitness);
}

void SolutionFinder::adjustFitnessNetwork(const float & factor)
{
	fitnessNetwork->adjustNetwork(factor);
}
