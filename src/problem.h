#pragma once

#include <cstddef>

#include <nlohmann/json.hpp>

struct problem_t
{
	double timestep;
	std::size_t dims;
	std::size_t iterations;
	std::size_t agents_count;
	std::size_t agent_types_count;
	std::array<double, 3> domain_size;

	static problem_t parse_from_json(const nlohmann::json& json)
	{
		problem_t problem;
		problem.timestep = json.at("timestep").get<double>();
		problem.dims = json.at("dims").get<std::size_t>();
		problem.iterations = json.at("iterations").get<std::size_t>();
		problem.agents_count = json.at("agents_count").get<std::size_t>();
		problem.agent_types_count = json.at("agent_types_count").get<std::size_t>();
		problem.domain_size = json.at("domain_size").get<std::array<double, 3>>();
		return problem;
	}
};
