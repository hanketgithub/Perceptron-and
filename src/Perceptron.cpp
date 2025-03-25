#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "Perceptron.h"

using namespace std;

// Sigmoid activation function
float sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

// Derivative of sigmoid (for backprop)
float sigmoid_derivative(float x) {
  float s = sigmoid(x);
  return s * (1.0f - s);
}

Neuron::Neuron(int inputSize) {
  for (int i = 0; i < inputSize; i++) {
    weights.push_back( ((float) rand() / RAND_MAX) - 0.5f );
  }

  bias = ((float) rand() / RAND_MAX) - 0.5f;
}

float Neuron::forward(const vector<float> &input) {
  input_sum = bias;
  for (size_t i = 0; i < input.size(); i++) {
    input_sum += input[i] * weights[i];
  }
  float output = sigmoid(input_sum);
  return output;
}

float Neuron::getBias() { return this->bias; }

void Neuron::setBias(float v) { this->bias = v; }

float Neuron::getInputSum() { return this->input_sum; }

vector<float> &Neuron::getWeights() { return this->weights; }

float Neuron::getWeight(int idx) { return this->weights[idx]; }

void Neuron::setWeight(int idx, float v) { this->weights[idx] = v; }

void Neuron::reset() {
  for (size_t i = 0; i < weights.size(); i++) {
    weights[i] = ((float) rand() / RAND_MAX) - 0.5f;
  }

  bias = ((float) rand() / RAND_MAX) - 0.5f;
  input_sum = 0;
}

void train(vector<vector<float>> &X, vector<float> &Y, Neuron &hidden1, Neuron &hidden2, Neuron &output_neuron, int epochs, const float learning_rate) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < X.size(); i++) {
      vector<float> input = X[i];
      float y = Y[i];

      // Forward pass
      float h1_output = hidden1.forward(input);
      float h2_output = hidden2.forward(input);
      vector<float> hidden_outputs = { h1_output, h2_output };
      float pred = output_neuron.forward(hidden_outputs);

      // Compute Mean Squared Error (MSE) loss
      float error = pred - y;
      total_loss += error * error;

      // --- Backward pass: output layer ---
      float d_output = 2 * error * sigmoid_derivative(output_neuron.getInputSum());

      // Update weights for output neuron
      for (size_t j = 0; j < output_neuron.getWeights().size(); j++) {
        float new_weight = output_neuron.getWeight(j) - learning_rate * d_output * hidden_outputs[j];

        output_neuron.setWeight(j, new_weight);
      }

      // Update bias for output neuron
      output_neuron.setBias( output_neuron.getBias() - learning_rate * d_output );

      // --- Backward pass: hidden layer ---
      float d_hidden1 = d_output * output_neuron.getWeight(0) * sigmoid_derivative(hidden1.getInputSum());
      float d_hidden2 = d_output * output_neuron.getWeight(1) * sigmoid_derivative(hidden2.getInputSum());

      for (size_t j = 0; j < hidden1.getWeights().size(); j++) {
        float new_weight1 = hidden1.getWeight(j) - learning_rate * d_hidden1 * input[j];
        hidden1.setWeight(j, new_weight1);

        float new_weight2 = hidden2.getWeight(j) - learning_rate * d_hidden2 * input[j];
        hidden2.setWeight(j, new_weight2);
      }

      // Update biases for hidden layer
      hidden1.setBias( hidden1.getBias() - learning_rate * d_hidden1 );
      hidden2.setBias( hidden2.getBias() - learning_rate * d_hidden2 );
    }

    // Print average loss every 1000 epochs
    if (epoch % 1000 == 0) {
      cout << "Epoch " << epoch << ", Loss = " << total_loss / X.size() << endl;
    }
  }
}
