
#include "nn.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "activation.h"
#include "common.h"
#include "dataset.h"

int nn::accuracy(const xfloat* Y, const int dim) const
{
	xfloat max_val = -2.0;
	int max_idx = 0;

	for (int i = 0; i < dim; i += 1)
	{
		const auto aa = a[layers.size() - 1];

		if (aa[i] > max_val)
		{
			max_val = aa[i];
			max_idx = i;
		}
	}

	return Y[max_idx] > 0.9 ? 1 : 0;
}

void nn::back_propagation(const xfloat* Y) const
{
	const auto layers_size = layers.size();
	const auto layer1 = layers[layers_size - 1];
	const auto layer2 = layers[layers_size - 2];
	const auto aa1 = a[layers_size - 1];
	const auto weight2 = weights[layers_size - 2];
	const auto delta3 = delta[layers_size - 3];
	const auto aa2 = a[layers_size - 2];
	const auto delta2 = delta[layers_size - 2];

	for (int neuron = 0; neuron < layer1; neuron += 1)
	{
		delta2[neuron] = (aa1[neuron] - Y[neuron]) * sig_derivative(aa1[neuron]);
	}

	for (int synapse = 0; synapse < layer2; synapse += 1)
	{
		xfloat REGISTER = 0.0;
		for (int neuron = 0; neuron < layer1; neuron += 1)
		{
			REGISTER += weight2[neuron][synapse] * delta2[neuron];
		}
		delta3[synapse] = REGISTER;
	}

	for (int synapse = 0; synapse < layer2; synapse += 1)
	{
		delta3[synapse] = delta3[synapse] * sig_derivative(aa2[synapse]);
	}

	for (int layer = 2; layer < layers_size - 1; layer += 1)
	{
		const auto dd1 = delta[layers_size - layer - 1];
		const auto dd2 = delta[layers_size - layer - 2];
		const auto aa1 = a[layers_size - layer - 1];
		const auto layer0 = layers[layers_size - layer];
		const auto layer1 = layers[layers_size - layer - 1];
		const auto ww1 = weights[layers_size - layer - 1];

		for (int synapse = 0; synapse < layer1; synapse += 1)
		{
			xfloat REGISTER = 0.0;

			for (int neuron = 0; neuron < layer0 - 1; neuron += 1) {
				REGISTER += ww1[neuron][synapse] * dd1[neuron];
			}
			dd2[synapse] = REGISTER;
		}

		for (int synapse = 0; synapse < layer1; synapse += 1)
		{
			dd2[synapse] = dd2[synapse] * sig_derivative(aa1[synapse]);
		}
	}
}

void nn::optimize(void) const
{
	const auto layers_size = layers.size();
	const auto weight2 = weights[layers_size - 2];
	const auto delta2 = delta[layers_size - 2];
	const auto aa2 = a[layers_size - 2];

	for (int neuron = 0; neuron < layers[layers_size - 1]; neuron += 1)
	{
		const auto nn = weight2[neuron];
		const auto dd = delta2[neuron];

		for (int synapse = 0; synapse < layers[layers_size - 2]; synapse += 1)
		{
			nn[synapse] -= LEARNING_RATE * dd * aa2[synapse];
		}
	}

	for (int layer = 2; layer < layers_size; layer += 1)
	{
		const auto weight1 = weights[layers_size - layer - 1];
		const auto delta1 = delta[layers_size - layer - 1];
		const auto layer0 = layers[layers_size - layer];
		const auto layer1 = layers[layers_size - layer - 1];
		const auto aa1 = a[layers_size - layer - 1];

		for (int neuron = 0; neuron < layer0 - 1; neuron += 1)
		{
			const auto ww = weight1[neuron];
			const auto dd = delta1[neuron];

			for (int synapse = 0; synapse < layer1; synapse += 1)
			{
				ww[synapse] -= LEARNING_RATE * dd * aa1[synapse];
			}
		}
	}
}

xfloat nn::mse_loss(const xfloat* Y, const int dim) const
{
	xfloat l = 0.0f;
	const auto* aa = a[layers.size() - 1];

	for (int i = 0; i < dim; i += 1)
	{
		l += (1.0f / 2.0f) * (Y[i] - aa[i]) * (Y[i] - aa[i]);
	}

	return l;
}

void nn::forward(void) const
{
	const auto layer_count = layers.size();

	for (int layer = 1; layer < layer_count - 1; layer += 1)
	{
		const auto weight1 = weights[layer - 1];
		const auto layer0 = layers[layer];
		const auto layer1 = layers[layer - 1];
		const auto zz = z[layer];
		const auto aa = a[layer];
		const auto aa2 = a[layer - 1];

		for (int neuron = 0; neuron < layer0 - 1; neuron += 1) {
			const auto ww = weight1[neuron];
			xfloat REGISTER = 0.0;

			for (int synapse = 0; synapse < layer1; synapse += 1) {
				REGISTER += ww[synapse] * aa2[synapse];
			}
			zz[neuron] = REGISTER;
		}

		for (int neuron = 0; neuron < layer0 - 1; neuron += 1)
		{
			aa[neuron] = sigmoid(zz[neuron]);
		}
	}

	const auto lc2 = layers[layer_count - 1];
	const auto zz2 = z[layer_count - 1];
	const auto aa1 = a[layer_count - 1];

	for (int neuron = 0; neuron < lc2; neuron += 1)
	{
		const auto layer2 = layers[layer_count - 2];
		const auto weight2 = weights[layer_count - 2];
		const auto ww = weight2[neuron];
		const auto aa2 = a[layer_count - 2];

		xfloat REGISTER = 0.0;

		for (int synapse = 0; synapse < layer2; synapse += 1)
		{
			REGISTER += ww[synapse] * aa2[synapse];
		}
		zz2[neuron] = REGISTER;
	}

	for (int neuron = 0; neuron < lc2; neuron += 1)
	{
		aa1[neuron] = sigmoid(zz2[neuron]);
	}
}


void nn::fit(const dataset& data) const
{
	const size_t sample_count = data.samples();

	std::array<xfloat, EPOCHS> loss;
	std::array<int, EPOCHS> validity;
	std::random_device rd;
	std::mt19937 gen(rd());	
	std::uniform_int_distribution<> dist(0, sample_count - 1);
	for (int epoch = 0; epoch < EPOCHS; epoch += 1) {
		loss[epoch] = 0.0;
		validity[epoch] = 0;
		for (int sample = 0; sample < sample_count; sample += 1) {
			const int shuffled_idx = dist(gen);
			zero_grad(data.X[shuffled_idx]);
			forward();
			back_propagation(data.Y[shuffled_idx]);
			optimize();
			loss[epoch] += mse_loss(data.Y[shuffled_idx], data.classes);
			validity[epoch] += accuracy(data.Y[shuffled_idx], data.classes);
		}
		const time_t end = time(nullptr);
		loss[epoch] /= (sample_count + 0.0f);

		const auto epoch_loss = loss[epoch];
		const auto epoch_accuracy = validity[epoch];
		std::cout << "\n[EPOCH " << std::setw(4) << epoch + 1 << "] [LOSS " << std::fixed << std::setprecision(5) << epoch_loss << "] [ACCURACY " << std::setw(6) << epoch_accuracy << " out of " << MNIST_TRAIN << "]";
	}
}

void nn::evaluate(const dataset& data)
{
	int validity = 0;
	xfloat loss = 0.0;
	const auto sample_count = data.samples();

	for (int sample = 0; sample < sample_count; sample += 1) {
		zero_grad(data.X[sample]);
		forward();
		loss += mse_loss(data.Y[sample], data.classes);
		validity += accuracy(data.Y[sample], data.classes);
	}
	loss /= (sample_count + 0.0f);

	std::cout << "\n\n[EVALUATION] [LOSS " << std::fixed << std::setprecision(5) << loss << "] [ACCURACY " << std::setw(6) << validity << " out of " << MNIST_TEST << "]";
}

int nn::get_label(const xfloat* y_pred) const
{
	int label = 0;
	xfloat max_val = -2.0;

	for (int i = 0; i < layers[layers.size() - 1]; i += 1)
	{
		if (y_pred[i] > max_val)
		{
			max_val = y_pred[i];
			label = i;
		}
	}

	return label;
}

int nn::predict(const xfloat* X) const
{
	zero_grad(X);
	forward();
	return get_label(a[layers.size() - 1]);
}

void nn::set_layers(const std::vector<int>& l)
{
	for (auto& elem : l)
	{
		layers.push_back(elem);
	}
}

void nn::set_z(const std::vector<int>& l)
{
	z = new xfloat * [l.size()];
	for (int i = 0; i < l.size(); i += 1)
	{
		z[i] = new xfloat[l[i]];
	}
}

void nn::set_a(const std::vector<int>& l)
{
	a = new xfloat * [l.size()];
	for (int i = 0; i < l.size(); i += 1)
	{
		a[i] = new xfloat[l[i]];
	}
}

void nn::set_delta(const std::vector<int>& l)
{
	delta = new xfloat * [l.size() - 1];
	for (int i = 1; i < l.size(); i += 1)
	{
		delta[i - 1] = new xfloat[l[i]];
	}
}

void nn::set_weights(const std::vector<int>& l, const xfloat min, const xfloat max)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(min, max);
	weights = new xfloat * *[l.size() - 1];
	for (int i = 1; i < l.size() - 1; i += 1)
	{
		weights[i - 1] = new xfloat * [l[i] - 1];
		for (int j = 0; j < l[i] - 1; j += 1)
		{
			weights[i - 1][j] = new xfloat[l[i - 1]];
			for (int k = 0; k < l[i - 1]; k += 1)
			{
				weights[i - 1][j][k] = dist(gen);
			}
		}
	}
	weights[l.size() - 2] = new xfloat * [l[l.size() - 1]];
	for (int j = 0; j < l[l.size() - 1]; j += 1) {
		weights[l.size() - 2][j] = new xfloat[l[l.size() - 2]];
		for (int k = 0; k < l[l.size() - 2]; k += 1)
		{
			weights[l.size() - 2][j][k] = dist(gen);
		}
	}
}

void nn::compile(const std::vector<int>& l, const xfloat min, const xfloat max)
{
	set_layers(l);
	set_z(l);
	set_a(l);
	set_delta(l);
	set_weights(l, min, max);
}

void nn::zero_grad(const xfloat* X) const
{
	for (int j = 0; j < layers[0] - 1; j += 1) {
		z[0][j] = X[j];
		a[0][j] = X[j];
	}
	z[0][layers[0] - 1] = 1.0;
	a[0][layers[0] - 1] = 1.0;

	for (int i = 1; i < layers.size() - 1; i += 1) {
		for (int j = 0; j < layers[i] - 1; j += 1)
		{
			z[i][j] = 0.0;
			a[i][j] = 0.0;
			delta[i - 1][j] = 0.0;
		}
		z[i][layers[i] - 1] = 1.0;
		a[i][layers[i] - 1] = 1.0;
		delta[i - 1][layers[i] - 1] = 0.0;
	}

	for (int j = 0; j < layers[layers.size() - 1]; j += 1) {
		z[layers.size() - 1][j] = 0.0;
		a[layers.size() - 1][j] = 0.0;
		delta[layers.size() - 2][j] = 0.0;
	}
}

void nn::summary(void) const
{
	int l = 0;

	std::cout << "\n\nNeural Network Summary:\t\t[f := Sigmoid]" << std::endl << std::endl;

	for (auto& elem : layers)
	{
		std::cout << "Layer " << ++l << "\t" << std::setw(4) << elem << " neurons\n";
	}
}
