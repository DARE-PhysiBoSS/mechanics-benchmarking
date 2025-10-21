#include "base_solver.h"

#include <iostream>

#include "../agent_distributor.h"
#include "solver_helper.h"

template <std::size_t dims, bool use_symmetry, typename index_t, typename real_t>
static constexpr void solve_pair(bool try_skip_repulsion, bool try_skip_adhesion, index_t lhs, index_t rhs,
								 index_t agent_types_count, real_t* __restrict__ velocity,
								 const real_t* __restrict__ position, const real_t* __restrict__ radius,
								 const real_t* __restrict__ repulsion_coeff, const real_t* __restrict__ adhesion_coeff,
								 const real_t* __restrict__ relative_maximum_adhesion_distance,
								 const real_t* __restrict__ adhesion_affinity, const index_t* __restrict__ agent_type)
{
	real_t position_difference[dims];

	const real_t distance = std::max<real_t>(position_helper<dims>::difference_and_distance(
												 position + lhs * dims, position + rhs * dims, position_difference),
											 0.00001);

	// compute repulsion
	real_t repulsion;
	if (try_skip_repulsion)
	{
		repulsion = 0;
		const real_t repulsive_distance = radius[lhs] + radius[rhs];

		if (distance < repulsive_distance)
		{
			repulsion = 1 - distance / repulsive_distance;

			repulsion *= repulsion;

			repulsion *= std::sqrt(repulsion_coeff[lhs] * repulsion_coeff[rhs]);
		}
	}
	else
	{
		const real_t repulsive_distance = radius[lhs] + radius[rhs];

		repulsion = 1 - distance / repulsive_distance;

		repulsion = repulsion < 0 ? 0 : repulsion;

		repulsion *= repulsion;

		repulsion *= std::sqrt(repulsion_coeff[lhs] * repulsion_coeff[rhs]);
	}

	// compute adhesion
	real_t adhesion;
	if (try_skip_adhesion)
	{
		adhesion = 0;
		const real_t adhesion_distance = relative_maximum_adhesion_distance[lhs] * radius[lhs]
										 + relative_maximum_adhesion_distance[rhs] * radius[rhs];

		if (distance < adhesion_distance)
		{
			adhesion = 1 - distance / adhesion_distance;

			adhesion *= adhesion;

			const index_t lhs_type = agent_type[lhs];
			const index_t rhs_type = agent_type[rhs];

			adhesion *= std::sqrt(adhesion_coeff[lhs] * adhesion_coeff[rhs]
								  * adhesion_affinity[lhs * agent_types_count + rhs_type]
								  * adhesion_affinity[rhs * agent_types_count + lhs_type]);
		}
	}
	else
	{
		const real_t adhesion_distance = relative_maximum_adhesion_distance[lhs] * radius[lhs]
										 + relative_maximum_adhesion_distance[rhs] * radius[rhs];

		adhesion = 1 - distance / adhesion_distance;

		adhesion = adhesion < 0 ? 0 : adhesion;

		adhesion *= adhesion;

		const index_t lhs_type = agent_type[lhs];
		const index_t rhs_type = agent_type[rhs];

		adhesion *=
			std::sqrt(adhesion_coeff[lhs] * adhesion_coeff[rhs] * adhesion_affinity[lhs * agent_types_count + rhs_type]
					  * adhesion_affinity[rhs * agent_types_count + lhs_type]);
	}

	real_t force = (repulsion - adhesion) / distance;

	if constexpr (use_symmetry)
	{
		position_helper<dims>::update_velocities_atomic(velocity + lhs * dims, velocity + rhs * dims,
														position_difference, force);
	}
	else
	{
		position_helper<dims>::update_velocity(velocity + lhs * dims, position_difference, force);
	}
}

template <typename real_t>
void base_solver<real_t>::solve()
{
	if (!use_symmetry_)
	{
#pragma omp parallel for schedule(static)
		for (index_t i = 0; i < agents_count_; i++)
		{
			for (index_t j = 0; j < agents_count_; j++)
			{
				if (i == j)
					continue;

				if (dims_ == 1)
					solve_pair<1, false>(try_skip_repulsion_, try_skip_adhesion_, i, j, agent_types_count_,
										 velocities_.get(), positions_.get(), radius_.get(), repulsion_coeff_.get(),
										 adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
										 agent_types_.get());
				else if (dims_ == 2)
					solve_pair<2, false>(try_skip_repulsion_, try_skip_adhesion_, i, j, agent_types_count_,
										 velocities_.get(), positions_.get(), radius_.get(), repulsion_coeff_.get(),
										 adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
										 agent_types_.get());
				else if (dims_ == 3)
					solve_pair<3, false>(try_skip_repulsion_, try_skip_adhesion_, i, j, agent_types_count_,
										 velocities_.get(), positions_.get(), radius_.get(), repulsion_coeff_.get(),
										 adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
										 agent_types_.get());
			}
		}
	}
	else
	{
#pragma omp parallel for
		for (index_t i = 0; i < agents_count_; i++)
		{
			for (index_t j = i + 1; j < agents_count_; j++)
			{
				if (dims_ == 1)
					solve_pair<1, true>(try_skip_repulsion_, try_skip_adhesion_, i, j, agent_types_count_,
										velocities_.get(), positions_.get(), radius_.get(), repulsion_coeff_.get(),
										adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
										agent_types_.get());
				else if (dims_ == 2)
					solve_pair<2, true>(try_skip_repulsion_, try_skip_adhesion_, i, j, agent_types_count_,
										velocities_.get(), positions_.get(), radius_.get(), repulsion_coeff_.get(),
										adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
										agent_types_.get());
				else if (dims_ == 3)
					solve_pair<3, true>(try_skip_repulsion_, try_skip_adhesion_, i, j, agent_types_count_,
										velocities_.get(), positions_.get(), radius_.get(), repulsion_coeff_.get(),
										adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
										agent_types_.get());
			}
		}
	}

// Update positions based on velocities
#pragma omp parallel for schedule(static)
	for (index_t i = 0; i < agents_count_; i++)
	{
		for (index_t d = 0; d < dims_; d++)
		{
			positions_[i * dims_ + d] += velocities_[i * dims_ + d] * timestep_;
			velocities_[i * dims_ + d] = 0;
		}
	}
}

template <typename real_t>
void base_solver<real_t>::initialize(const nlohmann::json& params, const problem_t& problem)
{
	use_symmetry_ = params.value("use_symmetry", false);
	try_skip_repulsion_ = params.value("try_skip_repulsion", false);
	try_skip_adhesion_ = params.value("try_skip_adhesion", false);

	dims_ = static_cast<index_t>(problem.dims);
	timestep_ = static_cast<real_t>(problem.timestep);
	agents_count_ = static_cast<index_t>(problem.agents_count);
	agent_types_count_ = static_cast<index_t>(problem.agent_types_count);

	// Initialize agent distributor
	agent_distributor<real_t> distributor(problem);

	// Access distributed agent data
	positions_ = std::move(distributor.positions_);
	velocities_ = std::move(distributor.velocities_);
	radius_ = std::move(distributor.radius_);
	repulsion_coeff_ = std::move(distributor.repulsion_coeff_);
	adhesion_coeff_ = std::move(distributor.adhesion_coeff_);
	max_adhesion_distance_ = std::move(distributor.max_adhesion_distance_);
	adhesion_affinity_ = std::move(distributor.adhesion_affinity_);
	agent_types_ = std::move(distributor.agent_types_);
}

template <typename real_t>
std::array<double, 3> base_solver<real_t>::access_agent(std::size_t agent_id)
{
	// Access agent data
	std::array<double, 3> agent_data = { 0.0, 0.0, 0.0 };
	for (std::size_t d = 0; d < static_cast<std::size_t>(dims_); d++)
	{
		agent_data[d] = static_cast<double>(positions_[agent_id * dims_ + d]);
	}
	return agent_data;
}

template <typename real_t>
void base_solver<real_t>::save(std::ostream& os) const
{
	// Save agent data to output stream
	for (std::size_t i = 0; i < static_cast<std::size_t>(agents_count_); i++)
	{
		os << "Agent " << i << ": ";
		for (std::size_t d = 0; d < static_cast<std::size_t>(dims_); d++)
		{
			os << positions_[i * dims_ + d] << " ";
		}
		os << std::endl;
	}
}

template class base_solver<float>;
template class base_solver<double>;
