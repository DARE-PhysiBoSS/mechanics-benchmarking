#include "algorithms.h"

#include <chrono>
#include <fstream>
#include <iostream>

#include "mechanics_solver.h"
#include "problem.h"
#include "solvers/base_solver.h"
#include "solvers/grid_solver.h"
#include "solvers/reference_solver.h"
#include "solvers/transposed_grid_solver.h"
#include "solvers/transposed_solver.h"

template <typename real_t>
std::map<std::string, std::function<std::unique_ptr<mechanics_solver>()>> get_solvers_map()
{
	std::map<std::string, std::function<std::unique_ptr<mechanics_solver>()>> solvers;

	solvers["ref"] = []() { return std::make_unique<reference_solver<real_t>>(); };
	solvers["base"] = []() { return std::make_unique<base_solver<real_t>>(); };
	solvers["transposed"] = []() { return std::make_unique<transposed_solver<real_t>>(); };
	solvers["transposed_grid"] = []() { return std::make_unique<transposed_grid_solver<real_t>>(); };
	solvers["grid"] = []() { return std::make_unique<grid_solver<real_t>>(); };

	return solvers;
}

algorithms::algorithms(bool double_precision, bool verbose) : double_precision_(double_precision), verbose_(verbose)
{
	if (double_precision)
		solvers_ = get_solvers_map<double>();
	else
		solvers_ = get_solvers_map<float>();
}

std::unique_ptr<mechanics_solver> algorithms::get_solver(const std::string& alg)
{
	auto solver = solvers_.at(alg)();
	return solver;
}

void algorithms::append_params(std::ostream& os, const nlohmann::json& params, bool header)
{
	std::vector<std::string> keys_to_skip = { "warmup_time", "outer_iterations", "inner_iterations", "benchmark_kind" };
	if (header)
	{
		for (auto it = params.begin(); it != params.end(); ++it)
		{
			if (std::find(keys_to_skip.begin(), keys_to_skip.end(), it.key()) == keys_to_skip.end())
				os << it.key() << ",";
		}
	}
	else
	{
		for (auto it = params.begin(); it != params.end(); ++it)
		{
			if (std::find(keys_to_skip.begin(), keys_to_skip.end(), it.key()) == keys_to_skip.end())
				os << it.value() << ",";
		}
	}
}

std::ofstream open_file(const std::string& file_name)
{
	std::ofstream file(file_name);
	if (!file.is_open())
		throw std::runtime_error("Cannot open file " + file_name);
	return file;
}

void algorithms::run(const std::string& alg, const problem_t& problem, const nlohmann::json& params,
					 std::optional<std::string> output_file)
{
	auto solver = get_solver(alg);

	solver->initialize(params, problem);

	if (output_file)
	{
		std::filesystem::path output_path(*output_file);
		auto init_output_path = (output_path.parent_path() / ("initial_" + output_path.filename().string()));

		auto init_output = open_file(init_output_path.string());
		solver->save(init_output);
	}

	solver->solve();

	if (output_file)
	{
		auto output = open_file(*output_file);
		solver->save(output);
	}
}

std::pair<double, double> algorithms::common_validate(mechanics_solver& alg, mechanics_solver& ref,
													  const problem_t& problem)
{
	double maximum_absolute_difference = 0.;
	double rmse = 0.;

	for (std::size_t i = 0; i < problem.agents_count; i++)
	{
		auto alg_agent = alg.access_agent(i);
		auto ref_agent = ref.access_agent(i);

		for (std::size_t d = 0; d < problem.dims; d++)
		{
			auto diff = std::abs(alg_agent[d] - ref_agent[d]);
			maximum_absolute_difference = std::max(maximum_absolute_difference, diff);
			rmse += diff * diff;

			auto relative_diff = diff / std::abs(ref_agent[d]);
			if (diff > absolute_difference_print_threshold_ && relative_diff > relative_difference_print_threshold_
				&& verbose_)
				std::cout << "At agent " << i << ", dimension " << d << ": " << alg_agent[d]
						  << " is not close to the reference " << ref_agent[d] << std::endl;
		}
	}

	rmse = std::sqrt(rmse / (problem.agents_count * problem.dims));

	return { maximum_absolute_difference, rmse };
}

void algorithms::validate(const std::string& alg, const problem_t& problem, const nlohmann::json& params)
{
	auto ref_solver = get_solver("ref");
	auto solver = get_solver(alg);

	ref_solver->initialize(params, problem);
	solver->initialize(params, problem);

	solver->solve();
	ref_solver->solve();

	auto [max_absolute_diff, rmse] = common_validate(*solver, *ref_solver, problem);

	std::cout << "Maximal absolute difference: " << max_absolute_diff << ", RMSE: " << rmse << std::endl;
}

void algorithms::benchmark(const std::string& alg, const problem_t& problem, const nlohmann::json& params)
{
	// make header
	{
		std::cout << "algorithm,precision,dims,iterations,n,";
		append_params(std::cout, params, true);

		std::cout << "time,std_dev" << std::endl;
	}

	// warmup
	{
		auto warmup_time_s = params.contains("warmup_time") ? (double)params["warmup_time"] : 3.0;
		auto start = std::chrono::steady_clock::now();
		auto end = start;
		do
		{
			auto solver = get_solver(alg);
			solver->initialize(params, problem);
			solver->solve();

			end = std::chrono::steady_clock::now();
		} while ((double)std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < warmup_time_s);
	}

	// measure
	{
		auto outer_iterations = params.contains("outer_iterations") ? (std::size_t)params["outer_iterations"] : 1;

		auto compute_mean_and_std = [](const std::vector<std::size_t>& times) {
			double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
			std::vector<double> variances(times.size());
			std::transform(times.begin(), times.end(), variances.begin(),
						   [mean](double x) { return (x - mean) * (x - mean); });
			double variance = std::accumulate(variances.begin(), variances.end(), 0.0) / times.size();
			double std_dev = std::sqrt(variance);
			return std::make_pair(mean, std_dev);
		};

		std::cout << alg << "," << (double_precision_ ? "D" : "S") << "," << problem.dims << "," << problem.iterations
				  << "," << problem.agents_count << ",";
		append_params(std::cout, params, false);

		std::vector<std::size_t> times;

		for (std::size_t i = 0; i < outer_iterations; i++)
		{
			auto solver = get_solver(alg);

			solver->initialize(params, problem);

			auto start = std::chrono::steady_clock::now();
			solver->solve();
			auto end = std::chrono::steady_clock::now();

			times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		}

		auto [mean, std_dev] = compute_mean_and_std(times);

		std::cout << std::fixed << std::setprecision(2);

		for (auto t : times)
		{
			if (verbose_)
				std::cout << t << " ";
		}

		std::cout << mean << "," << std_dev << std::endl;

		std::cout.unsetf(std::ios::floatfield);
	}
}
