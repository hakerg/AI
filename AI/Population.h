#pragma once
#include "NeuralNetwork.h"

class Population
{
	std::vector<NeuralNetwork> population;
	std::vector<float> fitnesses;

public:
	const size_t size;
	const std::vector<size_t> layers;

	Population(const std::vector<size_t> & _layers, const size_t & _size);
	~Population();
	NeuralNetwork generateCandidate(const float & mutationRate = 0.1f);
	void judgeCreature(const NeuralNetwork & network, const float & fitness);
};

