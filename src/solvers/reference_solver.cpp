#include "reference_solver.h"

#include "../agent_distributor.h"
#include "solver_helper.h"

template <std::size_t dims, typename index_t, typename real_t>
void solve_pair(index_t lhs, index_t rhs, index_t cell_defs_count, real_t* __restrict__ velocity,
				const real_t* __restrict__ position, const real_t* __restrict__ radius,
				const real_t* __restrict__ cell_cell_repulsion_strength,
				const real_t* __restrict__ cell_cell_adhesion_strength,
				const real_t* __restrict__ relative_maximum_adhesion_distance,
				const real_t* __restrict__ cell_adhesion_affinity, const index_t* __restrict__ cell_definition_index)
{
	real_t position_difference[dims];

	const real_t distance = std::max<real_t>(position_helper<dims>::difference_and_distance(
												 position + lhs * dims, position + rhs * dims, position_difference),
											 0.00001);

	// compute repulsion
	real_t repulsion;
	{
		const real_t repulsive_distance = radius[lhs] + radius[rhs];

		repulsion = 1 - distance / repulsive_distance;

		repulsion = repulsion < 0 ? 0 : repulsion;

		repulsion *= repulsion;

		repulsion *= std::sqrt(cell_cell_repulsion_strength[lhs] * cell_cell_repulsion_strength[rhs]);
	}

	// compute adhesion
	real_t adhesion;
	{
		const real_t adhesion_distance = relative_maximum_adhesion_distance[lhs] * radius[lhs]
										 + relative_maximum_adhesion_distance[rhs] * radius[rhs];

		adhesion = 1 - distance / adhesion_distance;

		adhesion *= adhesion;

		const index_t lhs_cell_def_index = cell_definition_index[lhs];
		const index_t rhs_cell_def_index = cell_definition_index[rhs];

		adhesion *= std::sqrt(cell_cell_adhesion_strength[lhs] * cell_cell_adhesion_strength[rhs]
							  * cell_adhesion_affinity[lhs * cell_defs_count + rhs_cell_def_index]
							  * cell_adhesion_affinity[rhs * cell_defs_count + lhs_cell_def_index]);
	}

	real_t force = (repulsion - adhesion) / distance;

	position_helper<dims>::update_velocity(velocity + lhs * dims, position_difference, force);
}

template <typename real_t>
void reference_solver<real_t>::solve()
{
	for (index_t i = 0; i < agents_count_; i++)
	{
		for (index_t j = 0; j < agents_count_; j++)
		{
			if (i == j)
				continue;

			if (dims_ == 1)
				solve_pair<1>(i, j, cell_definitions_count_, velocities_.get(), positions_.get(), radius_.get(),
							  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
							  adhesion_affinity_.get(), agent_types_.get());
			else if (dims_ == 2)
				solve_pair<2>(i, j, cell_definitions_count_, velocities_.get(), positions_.get(), radius_.get(),
							  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
							  adhesion_affinity_.get(), agent_types_.get());
			else if (dims_ == 3)
				solve_pair<3>(i, j, cell_definitions_count_, velocities_.get(), positions_.get(), radius_.get(),
							  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
							  adhesion_affinity_.get(), agent_types_.get());
		}
	}

	// Update positions based on velocities
	for (index_t i = 0; i < agents_count_; i++)
	{
		for (index_t d = 0; d < dims_; d++)
		{
			positions_[i * dims_ + d] += velocities_[i * dims_ + d];
		}
	}
}

template <typename real_t>
void reference_solver<real_t>::initialize(const nlohmann::json&, const problem_t& problem)
{
	dims_ = static_cast<index_t>(problem.dims);
	agents_count_ = static_cast<index_t>(problem.agents_count);

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
std::array<double, 3> reference_solver<real_t>::access_agent(std::size_t agent_id)
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
void reference_solver<real_t>::save(std::ostream& os) const
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

template class reference_solver<float>;
template class reference_solver<double>;
