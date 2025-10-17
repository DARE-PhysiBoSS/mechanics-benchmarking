#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "problem.h"

template <typename real_t>
class agent_distributor
{
public:
	enum class distribution_type
	{
		uniform,
		gaussian_groups
	};

	// SoA (Structure of Arrays) representation
	std::unique_ptr<real_t[]> positions_;
	std::unique_ptr<real_t[]> velocities_;
	std::unique_ptr<real_t[]> radius_;
	std::unique_ptr<real_t[]> repulsion_coeff_;
	std::unique_ptr<real_t[]> adhesion_coeff_;
	std::unique_ptr<real_t[]> max_adhesion_distance_;
	std::unique_ptr<real_t[]> adhesion_affinity_;
	std::unique_ptr<int32_t[]> agent_types_;

private:
	std::size_t agents_count_;
	std::size_t dims_;

public:
	// Constructor that takes a problem and distributes agents uniformly
	explicit agent_distributor(const problem_t& problem);

	// Constructor with distribution type selection
	agent_distributor(const problem_t& problem, distribution_type dist_type, std::size_t num_groups = 0,
					  real_t sigma = 1.0, unsigned int seed = 42);

	// Get the number of agents
	std::size_t get_agents_count() const { return agents_count_; }

	// Get the dimensionality
	std::size_t get_dims() const { return dims_; }

private:
	// Helper function to calculate grid dimensions for uniform distribution
	std::array<std::size_t, 3> calculate_grid_dimensions(const problem_t& problem) const;

	// Helper function to distribute agents uniformly in the domain
	void distribute_agents_uniformly(const problem_t& problem);

	// Helper function to distribute agents in Gaussian groups
	void distribute_agents_gaussian_groups(const problem_t& problem, std::size_t num_groups, real_t sigma,
										   unsigned int seed);

	// Helper function to initialize default physical properties
	void initialize_default_properties(const problem_t& problem);
};

// Template implementation
template <typename real_t>
agent_distributor<real_t>::agent_distributor(const problem_t& problem)
	: agents_count_(problem.agents_count), dims_(problem.dims)
{
	// Allocate memory for all SoA arrays
	positions_ = std::make_unique<real_t[]>(agents_count_ * dims_);
	velocities_ = std::make_unique<real_t[]>(agents_count_ * dims_);
	radius_ = std::make_unique<real_t[]>(agents_count_);
	repulsion_coeff_ = std::make_unique<real_t[]>(agents_count_);
	adhesion_coeff_ = std::make_unique<real_t[]>(agents_count_);
	max_adhesion_distance_ = std::make_unique<real_t[]>(agents_count_);
	adhesion_affinity_ = std::make_unique<real_t[]>(agents_count_);
	agent_types_ = std::make_unique<int32_t[]>(agents_count_);

	// Distribute agents uniformly across the domain
	distribute_agents_uniformly(problem);
}

template <typename real_t>
agent_distributor<real_t>::agent_distributor(const problem_t& problem, distribution_type dist_type,
											 std::size_t num_groups, real_t sigma, unsigned int seed)
	: agents_count_(problem.agents_count), dims_(problem.dims)
{
	// Allocate memory for all SoA arrays
	positions_ = std::make_unique<real_t[]>(agents_count_ * dims_);
	velocities_ = std::make_unique<real_t[]>(agents_count_ * dims_);
	radius_ = std::make_unique<real_t[]>(agents_count_);
	repulsion_coeff_ = std::make_unique<real_t[]>(agents_count_);
	adhesion_coeff_ = std::make_unique<real_t[]>(agents_count_);
	max_adhesion_distance_ = std::make_unique<real_t[]>(agents_count_);
	adhesion_affinity_ = std::make_unique<real_t[]>(agents_count_);
	agent_types_ = std::make_unique<int32_t[]>(agents_count_);

	// Choose distribution method
	if (dist_type == distribution_type::uniform)
	{
		distribute_agents_uniformly(problem);
	}
	else if (dist_type == distribution_type::gaussian_groups)
	{
		// If num_groups is 0, use agent_types_count as the number of groups
		std::size_t groups = (num_groups > 0) ? num_groups : problem.agent_types_count;
		distribute_agents_gaussian_groups(problem, groups, sigma, seed);
	}
}

template <typename real_t>
std::array<std::size_t, 3> agent_distributor<real_t>::calculate_grid_dimensions(const problem_t& problem) const
{
	std::array<std::size_t, 3> grid_dims = { 1, 1, 1 };

	// Calculate the number of agents per dimension to create a regular grid
	if (problem.dims == 1)
	{
		grid_dims[0] = problem.agents_count;
	}
	else if (problem.dims == 2)
	{
		// Try to create a square-ish grid
		grid_dims[0] = static_cast<std::size_t>(std::sqrt(problem.agents_count));
		grid_dims[1] = (problem.agents_count + grid_dims[0] - 1) / grid_dims[0];
	}
	else if (problem.dims == 3)
	{
		// Try to create a cube-ish grid
		grid_dims[0] = static_cast<std::size_t>(std::cbrt(problem.agents_count));
		std::size_t remaining = (problem.agents_count + grid_dims[0] - 1) / grid_dims[0];
		grid_dims[1] = static_cast<std::size_t>(std::sqrt(remaining));
		grid_dims[2] = (remaining + grid_dims[1] - 1) / grid_dims[1];
	}

	return grid_dims;
}

template <typename real_t>
void agent_distributor<real_t>::initialize_default_properties(const problem_t& problem)
{
	for (std::size_t i = 0; i < agents_count_; ++i)
	{
		// Initialize velocities to zero
		for (std::size_t d = 0; d < dims_; ++d)
		{
			velocities_[i * dims_ + d] = static_cast<real_t>(0.0);
		}

		// Set default physical properties
		radius_[i] = static_cast<real_t>(1.0);
		repulsion_coeff_[i] = static_cast<real_t>(1.0);
		adhesion_coeff_[i] = static_cast<real_t>(0.5);
		max_adhesion_distance_[i] = static_cast<real_t>(2.0);
		adhesion_affinity_[i] = static_cast<real_t>(1.0);

		// Distribute agent types uniformly
		agent_types_[i] = static_cast<int32_t>(i % problem.agent_types_count);
	}
}

template <typename real_t>
void agent_distributor<real_t>::distribute_agents_uniformly(const problem_t& problem)
{
	auto grid_dims = calculate_grid_dimensions(problem);

	// Calculate spacing between agents
	std::array<real_t, 3> spacing = {
		grid_dims[0] > 1 ? static_cast<real_t>(problem.domain_size[0]) / (grid_dims[0] - 1)
						 : static_cast<real_t>(problem.domain_size[0]) / 2,
		grid_dims[1] > 1 ? static_cast<real_t>(problem.domain_size[1]) / (grid_dims[1] - 1)
						 : static_cast<real_t>(problem.domain_size[1]) / 2,
		grid_dims[2] > 1 ? static_cast<real_t>(problem.domain_size[2]) / (grid_dims[2] - 1)
						 : static_cast<real_t>(problem.domain_size[2]) / 2
	};

	// Distribute agents on the grid
	std::size_t agent_idx = 0;
	for (std::size_t iz = 0; iz < grid_dims[2] && agent_idx < agents_count_; ++iz)
	{
		for (std::size_t iy = 0; iy < grid_dims[1] && agent_idx < agents_count_; ++iy)
		{
			for (std::size_t ix = 0; ix < grid_dims[0] && agent_idx < agents_count_; ++ix)
			{
				// Set position
				if (dims_ >= 1)
				{
					positions_[agent_idx * dims_ + 0] = ix * spacing[0];
				}
				if (dims_ >= 2)
				{
					positions_[agent_idx * dims_ + 1] = iy * spacing[1];
				}
				if (dims_ >= 3)
				{
					positions_[agent_idx * dims_ + 2] = iz * spacing[2];
				}

				++agent_idx;
			}
		}
	}

	// Initialize default properties
	initialize_default_properties(problem);
}

template <typename real_t>
void agent_distributor<real_t>::distribute_agents_gaussian_groups(const problem_t& problem, std::size_t num_groups,
																  real_t sigma, unsigned int seed)
{
	std::mt19937 rng(seed);
	std::normal_distribution<real_t> normal_dist(0.0, sigma);
	std::uniform_real_distribution<real_t> uniform_dist(0.0, 1.0);

	// Calculate group centers - distribute them uniformly across the domain
	std::vector<std::array<real_t, 3>> group_centers(num_groups);

	// Create a grid for group centers
	std::array<std::size_t, 3> center_grid_dims = { 1, 1, 1 };
	if (dims_ == 1)
	{
		center_grid_dims[0] = num_groups;
	}
	else if (dims_ == 2)
	{
		center_grid_dims[0] = static_cast<std::size_t>(std::sqrt(num_groups));
		center_grid_dims[1] = (num_groups + center_grid_dims[0] - 1) / center_grid_dims[0];
	}
	else if (dims_ == 3)
	{
		center_grid_dims[0] = static_cast<std::size_t>(std::cbrt(num_groups));
		std::size_t remaining = (num_groups + center_grid_dims[0] - 1) / center_grid_dims[0];
		center_grid_dims[1] = static_cast<std::size_t>(std::sqrt(remaining));
		center_grid_dims[2] = (remaining + center_grid_dims[1] - 1) / center_grid_dims[1];
	}

	// Calculate spacing between group centers to maximize separation
	real_t margin = static_cast<real_t>(3.0) * sigma; // Leave margin for Gaussian spread
	std::array<real_t, 3> center_spacing = {
		center_grid_dims[0] > 1
			? static_cast<real_t>(problem.domain_size[0] - 2 * margin) / static_cast<real_t>(center_grid_dims[0] - 1)
			: static_cast<real_t>(0.0),
		center_grid_dims[1] > 1
			? static_cast<real_t>(problem.domain_size[1] - 2 * margin) / static_cast<real_t>(center_grid_dims[1] - 1)
			: static_cast<real_t>(0.0),
		center_grid_dims[2] > 1
			? static_cast<real_t>(problem.domain_size[2] - 2 * margin) / static_cast<real_t>(center_grid_dims[2] - 1)
			: static_cast<real_t>(0.0)
	};

	// Place group centers
	std::size_t group_idx = 0;
	for (std::size_t iz = 0; iz < center_grid_dims[2] && group_idx < num_groups; ++iz)
	{
		for (std::size_t iy = 0; iy < center_grid_dims[1] && group_idx < num_groups; ++iy)
		{
			for (std::size_t ix = 0; ix < center_grid_dims[0] && group_idx < num_groups; ++ix)
			{
				group_centers[group_idx][0] = margin + ix * center_spacing[0];
				group_centers[group_idx][1] = margin + iy * center_spacing[1];
				group_centers[group_idx][2] = margin + iz * center_spacing[2];
				++group_idx;
			}
		}
	}

	// Distribute agents among groups
	std::size_t agents_per_group = agents_count_ / num_groups;
	std::size_t remaining_agents = agents_count_ % num_groups;

	std::size_t agent_idx = 0;
	for (std::size_t g = 0; g < num_groups; ++g)
	{
		std::size_t agents_in_this_group = agents_per_group + (g < remaining_agents ? 1 : 0);

		for (std::size_t a = 0; a < agents_in_this_group; ++a)
		{
			// Add Gaussian noise around group center
			for (std::size_t d = 0; d < dims_; ++d)
			{
				real_t position = group_centers[g][d] + normal_dist(rng);

				// Clamp to domain boundaries
				position =
					std::max(static_cast<real_t>(0.0), std::min(position, static_cast<real_t>(problem.domain_size[d])));

				positions_[agent_idx * dims_ + d] = position;
			}

			++agent_idx;
		}
	}

	// Initialize default properties
	initialize_default_properties(problem);
}
