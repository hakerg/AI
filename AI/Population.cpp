#include "Population.h"



Population::Population(const std::vector<size_t> & _layers, const size_t & _size) : layers(_layers), size(_size)
{
}


Population::~Population()
{
}

NeuralNetwork Population::generateCandidate(const float & mutationRate)
{
	if (population.size() != size)
	{
		return NeuralNetwork(layers);
	}
	else
	{
		size_t parent1 = rand() % size, parent2 = rand() % size;
		NeuralNetwork child(population[parent1]);
		child.combine(population[parent2]);
		child.mutate(mutationRate);
		return child;
	}
}

void Population::judgeCreature(const NeuralNetwork & network, const float & fitness)
{
	if (size != population.size())
	{
		population.push_back(network);
		fitnesses.push_back(fitness);
	}
	else
	{
		size_t weakestIndex = 0;
		for (size_t i = 1; i < size; i++)
		{
			if (fitnesses[i] < fitnesses[weakestIndex])
			{
				weakestIndex = i;
			}
		}
		population[weakestIndex] = network;
		fitnesses[weakestIndex] = fitness;
	}
}
