#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "Perceptron.h"

using namespace std;

int main() {
  const int epochs = 1000000;
  const float learning_rate = 0.1f;

  vector<vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  // expected AND output
  vector<float> Y = {0, 0, 0, 1};

  Neuron hidden1(2);
  Neuron hidden2(2);
  Neuron output_neuron(2);

  train(X, Y, hidden1, hidden2, output_neuron, epochs, learning_rate);

  // --- Inference after training ---
  cout << "\nTrained AND gate:\n";
  for (size_t i = 0; i < X.size(); ++i) {
    float h1_output = hidden1.forward(X[i]);
    float h2_output = hidden2.forward(X[i]);
    float prediction = output_neuron.forward({h1_output, h2_output});
    cout << X[i][0] << " AND " << X[i][1]
         << " = " << round(prediction)
         << " (raw: " << prediction << ")\n";
  }

  return 0;
}
