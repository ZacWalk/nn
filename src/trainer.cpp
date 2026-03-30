// Mini-batch SGD training loop and evaluation reporting for the neural network.

#include "trainer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

#include "dataset.h"
#include "nn.h"

std::vector<training_epoch_metrics> trainer::fit(nn& model, const dataset& data) const
{
	const std::size_t sample_count = data.samples();
	if (sample_count == 0)
	{
		std::cerr << "Error: Cannot train on an empty dataset" << std::endl;
		return {};
	}

	if (config.batch_size <= 0)
	{
		std::cerr << "Error: Batch size must be positive" << std::endl;
		return {};
	}

	std::vector<training_epoch_metrics> history(static_cast<std::size_t>(config.epochs));
	std::vector<std::size_t> indices(sample_count);
	std::iota(indices.begin(), indices.end(), 0);
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int epoch = 0; epoch < config.epochs; epoch += 1)
	{
		auto& epoch_metrics = history[static_cast<std::size_t>(epoch)];
		epoch_metrics.loss = 0.0f;
		epoch_metrics.correct = 0;
		std::shuffle(indices.begin(), indices.end(), gen);
		model.begin_batch();
		std::size_t batch_count = 0;

		for (const std::size_t sample_index : indices)
		{
			const auto sample_metrics = model.accumulate_gradients(
				data.sample_x(sample_index),
				data.sample_y(sample_index),
				data.classes);
			epoch_metrics.loss += sample_metrics.loss;
			epoch_metrics.correct += sample_metrics.correct;
			batch_count += 1;

			if (batch_count == static_cast<std::size_t>(config.batch_size))
			{
				model.apply_batch(batch_count);
				model.begin_batch();
				batch_count = 0;
			}
		}

		if (batch_count > 0)
		{
			model.apply_batch(batch_count);
		}

		epoch_metrics.loss /= static_cast<xfloat>(sample_count);
		std::cout << "\n[EPOCH " << std::setw(4) << epoch + 1
			<< "] [LOSS " << std::fixed << std::setprecision(5) << epoch_metrics.loss
			<< "] [ACCURACY " << std::setw(6) << epoch_metrics.correct
			<< " out of " << sample_count << "]";
	}

	return history;
}

evaluation_metrics trainer::evaluate(nn& model, const dataset& data) const
{
	evaluation_metrics metrics{};
	metrics.samples = data.samples();
	if (metrics.samples == 0)
	{
		std::cerr << "Error: Cannot evaluate an empty dataset" << std::endl;
		return metrics;
	}

	for (std::size_t sample_index = 0; sample_index < metrics.samples; sample_index += 1)
	{
		const auto sample_metrics = model.evaluate_sample(
			data.sample_x(sample_index),
			data.sample_y(sample_index),
			data.classes);
		metrics.loss += sample_metrics.loss;
		metrics.correct += sample_metrics.correct;
	}

	metrics.loss /= static_cast<xfloat>(metrics.samples);
	std::cout << "\n\n[EVALUATION] [LOSS " << std::fixed << std::setprecision(5) << metrics.loss
		<< "] [ACCURACY " << std::setw(6) << metrics.correct
		<< " out of " << metrics.samples << "]";
	return metrics;
}