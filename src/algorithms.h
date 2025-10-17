#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "mechanics_solver.h"
#include "problem.h"

class algorithms
{
	std::map<std::string, std::function<std::unique_ptr<mechanics_solver>()>> solvers_;

	bool double_precision_;
	bool verbose_;

	static constexpr double relative_difference_print_threshold_ = 0.001;
	static constexpr double absolute_difference_print_threshold_ = 1e-6;

	std::pair<double, double> common_validate(mechanics_solver& alg, mechanics_solver& ref, const problem_t& problem);

	void append_params(std::ostream& os, const nlohmann::json& params, bool header);

	std::unique_ptr<mechanics_solver> get_solver(const std::string& alg);

public:
	algorithms(bool double_precision, bool verbose);

	// Run the algorithm on the given problem for specified number of iterations
	void run(const std::string& alg, const problem_t& problem, const nlohmann::json& params,
			 std::optional<std::string> output_file);

	// Validate one iteration of the algorithm with the reference implementation
	void validate(const std::string& alg, const problem_t& problem, const nlohmann::json& params);

	// Measure the algorithm performance
	void benchmark(const std::string& alg, const problem_t& problem, const nlohmann::json& params);
};
