#pragma once

#include <cstddef>

#include <nlohmann/json.hpp>

#include "problem.h"

class mechanics_solver
{
public:
	virtual void solve() = 0;

	virtual void initialize(const nlohmann::json& params, const problem_t& problem) = 0;

	virtual std::array<double, 3> access_agent(std::size_t agent_id) = 0;

	virtual void save(std::ostream& os) const = 0;

	virtual ~mechanics_solver() = default;
};
