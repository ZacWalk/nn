
#include "dataset.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>


void dataset::read_csv(const char* filename, int dataset_flag, xfloat x_max)
{
	std::ifstream stream(filename);
	constexpr char delimiter[] = ",";
	size_t len = 0;

	if (!stream.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return;
	}

	int columns = 0;
	std::cout << "Loading " << filename << std::endl;

	std::string line;

	if (std::getline(stream, line))
	{
		char* xline = (char*)line.c_str();
		char* token = strtok(xline, delimiter);

		while (token != nullptr)
		{
			token = strtok(nullptr, delimiter);
			columns += 1;
		}
	}

	dimensions = columns - 1;

	while (std::getline(stream, line))
	{
		char* xline = (char*)line.c_str();
		char* token = strtok(xline, delimiter);

		columns = 0;
		
		auto xx = new xfloat[dimensions];
		auto yy = new xfloat[classes];

		const auto label_val = std::stoi(token);

		for (int y_idx = 0; y_idx < classes; y_idx += 1)
		{
			if (y_idx == label_val)
			{
				yy[y_idx] = 1.0;
			}
			else
			{
				yy[y_idx] = 0.0;
			}
		}

		while (token != nullptr)
		{
			token = strtok(nullptr, delimiter);
			if (token != nullptr)
			{
				const auto pixel = std::stoi(token);
				xx[columns] = (pixel + 0.0) / x_max;
				columns += 1;
			}
		}

		X.emplace_back(xx);
		Y.emplace_back(yy);
	}
}
