#include "NeuralNetwork.h"



void NeuralNetwork::calculate()
{
	if (calculated) return;
	for (auto it = neurons.begin() + 1; it != neurons.end(); it++)
	{
		for (Neuron& n : *it)
		{
			float& v = n.input;
			v = 0.0f;
			auto& prevLayer = *(it - 1);
			for (size_t i = 0; i < n.inputSize; i++)
			{
				v += prevLayer[i].output * n.inputWeights[i];
			}
			v += n.bias;
			n.calculateOutput();
		}
	}
	calculated = true;
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t> & layers) : inputSize(layers.front()), outputSize(layers.back())
{
	neurons.push_back(std::vector<Neuron>(inputSize));
	for (auto it = layers.begin() + 1; it != layers.end(); it++)
	{
		neurons.push_back(std::vector<Neuron>(*it, Neuron(*(it - 1))));
	}
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::calculateGradient(const float * expectedOutput)
{
	calculate();
	for (Neuron& o : neurons.back())
	{
		if (*expectedOutput < 0.0f || *expectedOutput > 1.0f) throw std::string("Output should be within range from 0.0f to 1.0f");
		o.dC_do = o.output - *expectedOutput;
		++expectedOutput;
	}
	for (auto it = neurons.end() - 1; it != neurons.begin(); --it)
	{
		auto& prevLayer = *(it - 1);
		for (auto& pn : prevLayer)
		{
			pn.dC_do = 0.0f;
		}
		for (Neuron& n : *it)
		{
			n.dC_dib = n.dC_do * n.get_do_di();
			n.dC_dibTotal += n.dC_dib;
			for (size_t w = 0; w < n.inputSize; w++)
			{
				auto& prevNeuron = prevLayer[w];
				n.dC_dw[w] += n.dC_dib * prevNeuron.output;
				prevNeuron.dC_do += n.dC_dib * n.inputWeights[w];
			}
		}
	}
	gradients++;
}

float NeuralNetwork::getError(const float * expectedOutput)
{
	float error = 0.0f;
	for (Neuron& o : neurons.back())
	{
		error += o.dC_do * o.dC_do;
		++expectedOutput;
	}
	return error;
}

float NeuralNetwork::getLastError()
{
	float error = 0.0f;
	for (Neuron& o : neurons.back())
	{
		error += o.dC_do * o.dC_do;
	}
	return error;
}

void NeuralNetwork::adjustNetwork(float factor)
{
	if (gradients == 0) throw std::string("Gradient not calculated");
	factor /= gradients;
	gradients = 0;
	for (auto it = neurons.begin() + 1; it != neurons.end(); it++)
	{
		for (auto& n : *it)
		{
			n.applyGradient(factor);
		}
	}
	calculated = false;
}

void NeuralNetwork::copyInput(const float * source)
{
	for (auto& v : neurons.front())
	{
		if (v.output != *source)
		{
			v.output = *source;
			calculated = false;
		}
		++source;
	}
}

void NeuralNetwork::getOutput(float * dest)
{
	calculate();
	for (auto& v : neurons.back())
	{
		*dest = v.output;
		++dest;
	}
}

void NeuralNetwork::getOutput(float * input, float * output)
{
	copyInput(input);
	getOutput(output);
}

void NeuralNetwork::mutate(const float & mutationRate)
{
	for (auto it = neurons.begin() + 1; it != neurons.end(); it++)
	{
		for (Neuron& n : *it)
		{
			n.mutate(mutationRate);
		}
	}
	calculated = false;
}

void NeuralNetwork::combine(const NeuralNetwork & nn)
{
	size_t layers = neurons.size();
	if (layers != nn.neurons.size()) throw std::string("Invalid argument");
	if (inputSize != nn.inputSize) throw std::string("Invalid argument");
	for (size_t l = 1; l < layers; l++)
	{
		auto& layer = neurons[l];
		auto& layer2 = nn.neurons[l];
		size_t size = layer.size();
		if (size != layer2.size()) throw std::string("Invalid argument");
		for (size_t n = 0; n < size; n++)
		{
			layer[n].combine(layer2[n]);
		}
	}
	calculated = false;
}

const std::vector<std::vector<Neuron>> & NeuralNetwork::getNeurons()
{
	return neurons;
}

NeuralNetwork & NeuralNetwork::operator=(const NeuralNetwork & nn)
{
	if (inputSize != nn.inputSize || outputSize != nn.outputSize) throw std::string("Invalid argument");
	neurons = nn.neurons;
	calculated = nn.calculated;
	gradients = nn.gradients;
	return *this;
}

void NeuralNetwork::findBestInput(const size_t & firstInputIndex, const size_t & inputCount, const int & attempts, const float & factor)
{
	if (outputSize != 1) throw std::string("Network should have one output");
	auto& inputLayer = neurons.front();
	auto firstIt = inputLayer.begin() + firstInputIndex;
	auto endIt = firstIt + inputCount;
	for (int n = 0; n < attempts; n++)
	{
		calculate();
		for (auto it = neurons.end() - 1; it != neurons.begin(); it++)
		{
			auto& prevLayer = *(it - 1);
			size_t prevLayerSize = prevLayer.size();
			for (auto& pn : prevLayer)
			{
				pn.df_do = 0.0f;
			}
			for (Neuron& n : *it)
			{
				n.df_di = n.df_do * n.get_do_di();
				size_t loopStart, loopEnd;
				if (it == neurons.begin() + 1)
				{
					loopStart = firstInputIndex;
					loopEnd = firstInputIndex + inputCount;
				}
				else
				{
					loopStart = 0;
					loopEnd = prevLayerSize;
				}
				for (size_t pn = loopStart; pn < loopEnd; pn++)
				{
					prevLayer[pn].df_do += n.df_di * n.inputWeights[pn];
				}
			}
		}
		for (auto it = firstIt; it != endIt; it++)
		{
			it->output += it->df_do * factor;
		}
	}
}
