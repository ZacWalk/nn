
// Multilayer perceptron implementation with forward pass, backpropagation, and AVX2-accelerated math helpers.

#include "nn.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "activation.h"

namespace
{
#if defined(__AVX2__)
xfloat horizontal_sum_avx(__m256 value)
{
	const __m128 low = _mm256_castps256_ps128(value);
	const __m128 high = _mm256_extractf128_ps(value, 1);
	__m128 sum128 = _mm_add_ps(low, high);
	sum128 = _mm_hadd_ps(sum128, sum128);
	sum128 = _mm_hadd_ps(sum128, sum128);
	return _mm_cvtss_f32(sum128);
}
#endif

xfloat dot_product(const xfloat* lhs, const xfloat* rhs, const int count)
{
	xfloat accumulator = 0.0f;
	int index = 0;

#if defined(__AVX2__)
	__m256 sum = _mm256_setzero_ps();
	for (; index + 8 <= count; index += 8)
	{
		const __m256 left = _mm256_loadu_ps(lhs + index);
		const __m256 right = _mm256_loadu_ps(rhs + index);
		sum = _mm256_add_ps(sum, _mm256_mul_ps(left, right));
	}
	accumulator = horizontal_sum_avx(sum);
#endif

	for (; index < count; index += 1)
	{
		accumulator += lhs[index] * rhs[index];
	}

	return accumulator;
}

void scaled_accumulate(xfloat* destination, const xfloat* input, const int count, const xfloat scale)
{
	int index = 0;

#if defined(__AVX2__)
	const __m256 scale_vector = _mm256_set1_ps(scale);
	for (; index + 8 <= count; index += 8)
	{
		const __m256 destination_vector = _mm256_loadu_ps(destination + index);
		const __m256 input_vector = _mm256_loadu_ps(input + index);
		const __m256 scaled_input = _mm256_mul_ps(scale_vector, input_vector);
		_mm256_storeu_ps(destination + index, _mm256_add_ps(destination_vector, scaled_input));
	}
#endif

	for (; index < count; index += 1)
	{
		destination[index] += scale * input[index];
	}
}

void apply_momentum_update(xfloat* weights, xfloat* velocity, xfloat* gradients, const int count, const xfloat momentum, const xfloat learning_rate)
{
	int index = 0;

#if defined(__AVX2__)
	const __m256 momentum_vector = _mm256_set1_ps(momentum);
	const __m256 learning_rate_vector = _mm256_set1_ps(learning_rate);
	const __m256 zero_vector = _mm256_setzero_ps();
	for (; index + 8 <= count; index += 8)
	{
		const __m256 weight_vector = _mm256_loadu_ps(weights + index);
		const __m256 velocity_vector = _mm256_loadu_ps(velocity + index);
		const __m256 gradient_vector = _mm256_loadu_ps(gradients + index);
		const __m256 updated_velocity = _mm256_sub_ps(_mm256_mul_ps(momentum_vector, velocity_vector), _mm256_mul_ps(learning_rate_vector, gradient_vector));
		_mm256_storeu_ps(velocity + index, updated_velocity);
		_mm256_storeu_ps(weights + index, _mm256_add_ps(weight_vector, updated_velocity));
		_mm256_storeu_ps(gradients + index, zero_vector);
	}
#endif

	for (; index < count; index += 1)
	{
		velocity[index] = momentum * velocity[index] - learning_rate * gradients[index];
		weights[index] += velocity[index];
		gradients[index] = 0.0f;
	}
}
}

const nn::layer_descriptor& nn::output_layer(void) const
{
	return layers.back();
}

std::size_t nn::parameter_count(void) const
{
	return weights.size();
}

xfloat* nn::activation_ptr(const std::size_t layer)
{
	return activations.data() + layers[layer].activation_offset;
}

const xfloat* nn::activation_ptr(const std::size_t layer) const
{
	return activations.data() + layers[layer].activation_offset;
}

xfloat* nn::delta_ptr(const std::size_t layer)
{
	return deltas.data() + layers[layer].delta_offset;
}

const xfloat* nn::delta_ptr(const std::size_t layer) const
{
	return deltas.data() + layers[layer].delta_offset;
}

xfloat* nn::weight_ptr(const std::size_t layer)
{
	return weights.data() + connections[layer].weight_offset;
}

const xfloat* nn::weight_ptr(const std::size_t layer) const
{
	return weights.data() + connections[layer].weight_offset;
}

xfloat* nn::gradient_ptr(const std::size_t layer)
{
	return gradient_accumulators.data() + connections[layer].weight_offset;
}

xfloat* nn::velocity_ptr(const std::size_t layer)
{
	return velocity.data() + connections[layer].weight_offset;
}

sample_metrics nn::evaluate_output(const xfloat* Y, const int dim, const bool write_output_delta)
{
	sample_metrics metrics{};
	const std::size_t last_layer = layers.size() - 1;
	const auto* output = activation_ptr(last_layer);
	auto* output_delta = write_output_delta ? delta_ptr(last_layer) : nullptr;
	int label = 0;
	int target_label = 0;
	xfloat max_val = output[0];

	for (int neuron = 0; neuron < dim; neuron += 1)
	{
		const xfloat prediction = output[neuron];
		const xfloat target = Y[neuron];
		const xfloat diff = prediction - target;
		if (target > 0.0f)
		{
			target_label = neuron;
		}
		if (write_output_delta)
		{
			output_delta[neuron] = diff;
		}
		if (prediction > max_val)
		{
			max_val = prediction;
			label = neuron;
		}
	}

	metrics.loss = -std::log(std::max(output[target_label], std::numeric_limits<xfloat>::min()));
	metrics.correct = Y[label] > 0.9f ? 1u : 0u;
	return metrics;
}

sample_metrics nn::train_sample(const xfloat* X, const xfloat* Y, const int dim)
{
	begin_batch();
	const auto metrics = accumulate_gradients(X, Y, dim);
	apply_batch(1);
	return metrics;
}

void nn::begin_batch(void)
{
	std::fill(gradient_accumulators.begin(), gradient_accumulators.end(), 0.0f);
}

sample_metrics nn::accumulate_gradients(const xfloat* X, const xfloat* Y, const int dim)
{
	load_input(X);
	forward();
	const auto metrics = evaluate_output(Y, dim, true);
	back_propagation();
	accumulate_weight_gradients();
	return metrics;
}

void nn::apply_batch(const std::size_t batch_size)
{
	if (batch_size == 0)
	{
		throw std::invalid_argument("Batch size must be positive");
	}

	const xfloat batch_learning_rate = learning_rate / static_cast<xfloat>(batch_size);
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		const auto& connection = connections[layer];
		auto* layer_weights = weight_ptr(layer);
		auto* layer_velocity = velocity_ptr(layer);
		auto* layer_gradients = gradient_ptr(layer);
		const std::size_t weight_count = static_cast<std::size_t>(connection.input_width) * static_cast<std::size_t>(connection.output_width);
		apply_momentum_update(layer_weights, layer_velocity, layer_gradients, static_cast<int>(weight_count), momentum, batch_learning_rate);
	}
}

sample_metrics nn::evaluate_sample(const xfloat* X, const xfloat* Y, const int dim)
{
	load_input(X);
	forward();
	return evaluate_output(Y, dim, false);
}

void nn::back_propagation(void)
{
	const std::size_t last_layer = layers.size() - 1;

	for (std::size_t layer = last_layer - 1; layer > 0; layer -= 1)
	{
		const auto& current_layer = layers[layer];
		const auto& next_connection = connections[layer];
		const auto* next_weights = weight_ptr(layer);
		const auto* next_delta = delta_ptr(layer + 1);
		const auto* current_activation = activation_ptr(layer);
		auto* current_delta = delta_ptr(layer);

		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			xfloat accumulator = 0.0f;
			for (int next_neuron = 0; next_neuron < next_connection.output_width; next_neuron += 1)
			{
				accumulator += next_weights[next_neuron * next_connection.input_width + neuron] * next_delta[next_neuron];
			}
			current_delta[neuron] = accumulator * sig_derivative(current_activation[neuron]);
		}
	}
}

void nn::accumulate_weight_gradients(void)
{
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		const auto& connection = connections[layer];
		auto* layer_gradients = gradient_ptr(layer);
		const auto* layer_input = activation_ptr(layer);
		const auto* layer_delta = delta_ptr(layer + 1);

		for (int neuron = 0; neuron < connection.output_width; neuron += 1)
		{
			auto* neuron_gradients = layer_gradients + neuron * connection.input_width;
			scaled_accumulate(neuron_gradients, layer_input, connection.input_width, layer_delta[neuron]);
		}
	}
}

void nn::forward(void)
{
	for (std::size_t layer = 1; layer < layers.size(); layer += 1)
	{
		const auto& current_layer = layers[layer];
		const auto& current_connection = connections[layer - 1];
		const auto* previous_activation = activation_ptr(layer - 1);
		auto* current_activation = activation_ptr(layer);
		const auto* layer_weights = weight_ptr(layer - 1);
		xfloat max_logit = std::numeric_limits<xfloat>::lowest();

		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			const auto* neuron_weights = layer_weights + neuron * current_connection.input_width;
			const xfloat accumulator = dot_product(neuron_weights, previous_activation, current_connection.input_width);

			if (current_layer.has_bias)
			{
				current_activation[neuron] = sigmoid(accumulator);
			}
			else
			{
				current_activation[neuron] = accumulator;
				max_logit = std::max(max_logit, accumulator);
			}
		}

		if (current_layer.has_bias)
		{
			current_activation[current_layer.width] = 1.0f;
			continue;
		}

		xfloat denominator = 0.0f;
		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			current_activation[neuron] = std::exp(current_activation[neuron] - max_logit);
			denominator += current_activation[neuron];
		}

		for (int neuron = 0; neuron < current_layer.width; neuron += 1)
		{
			current_activation[neuron] /= denominator;
		}
	}
}

int nn::get_label(const xfloat* y_pred) const
{
	int label = 0;
	xfloat max_val = y_pred[0];

	for (int i = 1; i < output_layer().width; i += 1)
	{
		if (y_pred[i] > max_val)
		{
			max_val = y_pred[i];
			label = i;
		}
	}

	return label;
}

int nn::predict(const xfloat* X)
{
	load_input(X);
	forward();
	return get_label(activation_ptr(layers.size() - 1));
}

void nn::compile(const nn_config& config)
{
	const auto& l = config.layers;
	if (l.size() < 2)
	{
		throw std::invalid_argument("Network must contain at least an input and output layer");
	}

	for (const int width : l)
	{
		if (width <= 0)
		{
			throw std::invalid_argument("Layer widths must be positive");
		}
	}

	if (config.learning_rate <= 0.0f)
	{
		throw std::invalid_argument("Learning rate must be positive");
	}

	if (config.momentum < 0.0f || config.momentum >= 1.0f)
	{
		throw std::invalid_argument("Momentum must be in [0, 1)");
	}

	layers.assign(l.size(), {});
	connections.assign(l.size() - 1, {});
	learning_rate = config.learning_rate;
	momentum = config.momentum;

	std::size_t activation_size = 0;
	for (std::size_t layer = 0; layer < layers.size(); layer += 1)
	{
		const bool has_bias = layer + 1 < l.size();
		layers[layer].width = l[layer];
		layers[layer].has_bias = has_bias;
		layers[layer].activation_width = l[layer] + (has_bias ? 1 : 0);
		layers[layer].activation_offset = activation_size;
		activation_size += static_cast<std::size_t>(layers[layer].activation_width);
	}
	activations.assign(activation_size, 0.0f);

	std::size_t delta_size = 0;
	for (std::size_t layer = 1; layer < layers.size(); layer += 1)
	{
		layers[layer].delta_offset = delta_size;
		delta_size += static_cast<std::size_t>(layers[layer].width);
	}
	deltas.assign(delta_size, 0.0f);

	std::size_t weight_size = 0;
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		connections[layer].input_width = layers[layer].activation_width;
		connections[layer].output_width = layers[layer + 1].width;
		connections[layer].weight_offset = weight_size;
		weight_size += static_cast<std::size_t>(connections[layer].output_width) * static_cast<std::size_t>(connections[layer].input_width);
	}
	weights.resize(weight_size);
	gradient_accumulators.assign(weight_size, 0.0f);
	velocity.assign(weight_size, 0.0f);

	std::random_device rd;
	std::mt19937 gen(rd());
	for (std::size_t layer = 0; layer < connections.size(); layer += 1)
	{
		const auto& connection = connections[layer];
		xfloat layer_min = config.weight_min;
		xfloat layer_max = config.weight_max;
		if (config.use_xavier_initialization)
		{
			const xfloat limit = std::sqrt(6.0f / static_cast<xfloat>(connection.input_width + connection.output_width));
			layer_min = -limit;
			layer_max = limit;
		}

		std::uniform_real_distribution<xfloat> dist(layer_min, layer_max);
		auto* layer_weights = weight_ptr(layer);
		const std::size_t layer_weight_count = static_cast<std::size_t>(connection.input_width) * static_cast<std::size_t>(connection.output_width);
		for (std::size_t weight_index = 0; weight_index < layer_weight_count; weight_index += 1)
		{
			layer_weights[weight_index] = dist(gen);
		}
	}
}

void nn::compile(const std::vector<int>& l, const xfloat min, const xfloat max)
{
	compile(nn_config{ l, LEARNING_RATE, 0.0f, false, min, max });
}

void nn::load_input(const xfloat* X)
{
	if (layers.empty())
	{
		throw std::logic_error("Network must be compiled before use");
	}

	auto* input = activation_ptr(0);
	std::copy_n(X, static_cast<std::size_t>(layers[0].width), input);
	if (layers[0].has_bias)
	{
		input[layers[0].width] = 1.0f;
	}
}

void nn::summary(void) const
{
	int layer_number = 0;

	std::cout << "\n\nNeural Network Summary:\t\t[hidden := Sigmoid, output := Softmax]" << std::endl << std::endl;

	for (std::size_t layer = 0; layer < layers.size(); layer += 1)
	{
		std::cout << "Layer " << ++layer_number << "\t" << std::setw(4) << layers[layer].width << " neurons";
		if (layers[layer].has_bias)
		{
			std::cout << " + bias";
		}
		std::cout << "\n";
	}

	std::cout << "Parameters\t" << parameter_count() << " trainable weights\n";
	std::cout << "Learning Rate\t" << learning_rate << "\n";
	std::cout << "Momentum\t" << momentum << "\n";
}