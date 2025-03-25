#ifndef PERCEPTRON_H
#define PERCEPTRON_H

class Neuron {
private:
  std::vector<float> weights;
  float input_sum;
  float bias;

public:
  Neuron(int inputSize);

  float forward(const std::vector<float> &input);

  float getBias();
  void setBias(float v);

  float getInputSum();

  std::vector<float> &getWeights();
  float               getWeight(int idx);
  void                setWeight(int idx, float v);

  void reset();
};


float sigmoid(float x);

float sigmoid_derivative(float x);

void train(std::vector<std::vector<float>> &X, std::vector<float> &Y, Neuron &hidden1, Neuron &hidden2, Neuron &output_neuron, int epochs, const float learning_rate);

#endif
