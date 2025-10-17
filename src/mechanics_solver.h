#pragma once

#include <cstddef>

class mechanics_solver
{
public:
	virtual void solve(std::size_t iterations) = 0;
	virtual ~mechanics_solver() = default;
};
