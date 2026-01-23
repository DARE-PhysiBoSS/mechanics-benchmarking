#include <fstream>
#include <iostream>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include "algorithms.h"
#include "perf_utils.h"
#include "problem.h"

// Helper function to load JSON from file
nlohmann::json load_json_file(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open file: " + filename);
	}
	nlohmann::json j;
	file >> j;
	return j;
}

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("mechanize", "1.0");

	// Add arguments
	program.add_argument("--alg").required().help("Algorithm to run");
	program.add_argument("--params").required().help("Path to JSON parameters file");
	program.add_argument("--problem").required().help("Path to JSON problem file");
	auto& group = program.add_mutually_exclusive_group();
	group.add_argument("--validate").help("Validate algorithm correctness").default_value(false).implicit_value(true);
	group.add_argument("--benchmark").help("Benchmark algorithm performance").default_value(false).implicit_value(true);
	group.add_argument("--run").nargs(0, 1).help("Run algorithm and optionally output to FILE");
	group.add_argument("--profile").help("Profile using PAPI counters.").default_value(false).implicit_value(true);
	program.add_argument("--verbose").help("Enable verbose output").default_value(false).implicit_value(true);
	program.add_argument("--double").help("Use double precision").default_value(false).implicit_value(true);

	// Parse arguments
	try
	{
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	// Extract parsed arguments
	std::string alg = program.get<std::string>("--alg");
	std::string params_file = program.get<std::string>("--params");
	std::string problem_file = program.get<std::string>("--problem");
	bool validate = program.get<bool>("--validate");
	bool benchmark = program.get<bool>("--benchmark");
	bool verbose = program.get<bool>("--verbose");
	bool profile = program.get<bool>("--profile");
	bool double_precision = program.get<bool>("--double");

	// Check if --run was provided
	bool run = program.is_used("--run");
	std::optional<std::string> output_file;
	if (run && program.present("--run"))
	{
		output_file = program.get<std::string>("--run");
	}

	// Validate that at least one operation is specified
	if (!validate && !benchmark && !run && !profile)
	{
		std::cerr << "Error: At least one of --validate, --benchmark, --run, or --profile must be specified" << std::endl;
		return 1;
	}

	try
	{
		// Load JSON files
		if (verbose)
		{
			std::cout << "Loading parameters from: " << params_file << std::endl;
			std::cout << "Loading problem from: " << problem_file << std::endl;
		}

		nlohmann::json params = load_json_file(params_file);
		nlohmann::json problem_json = load_json_file(problem_file);
		problem_t problem = problem_t::parse_from_json(problem_json);

		// Create algorithms instance
		// Note: Assuming double_precision is always true; adjust as needed
		algorithms algs(double_precision, verbose);

		if (verbose)
		{
			std::cout << "Algorithm: " << alg << std::endl;
			std::cout << "Problem - Timestep: " << problem.timestep << ", Agents: " << problem.agents_count
					  << ", Agent Types: " << problem.agent_types_count << std::endl;
		}

		// Execute requested operations
		if (validate)
		{
			if (verbose)
			{
				std::cout << "Validating algorithm..." << std::endl;
			}
			algs.validate(alg, problem, params);
		}

		if (benchmark)
		{
			if (verbose)
			{
				std::cout << "Benchmarking algorithm..." << std::endl;
			}
			algs.benchmark(alg, problem, params);
		}

		if (run)
		{
			if (verbose)
			{
				std::cout << "Running algorithm";
				if (output_file)
				{
					std::cout << " (output to: " << *output_file << ")";
				}
				std::cout << std::endl;
			}
			algs.run(alg, problem, params, output_file);
		}

		if (profile)
		{
			if (verbose)
			{
				std::cout << "Profiling algorithm..." << std::endl;
			}
			perf_counter::enable();
			algs.run(alg, problem, params, output_file);
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	if (verbose)
	{
		std::cout << "Completed successfully" << std::endl;
	}

	return 0;
}
