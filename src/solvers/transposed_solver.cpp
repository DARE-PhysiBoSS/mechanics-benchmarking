#include "transposed_solver.h"

#include <iostream>

#include "../agent_distributor.h"

template <std::size_t dims, typename index_t, typename real_t>
void solve_pair(index_t lhs_begin, index_t lhs_end, index_t agents_count, index_t agent_types_count,
				real_t* __restrict__ velocity_x, real_t* __restrict__ velocity_y, real_t* __restrict__ velocity_z,
				const real_t* __restrict__ position_x, const real_t* __restrict__ position_y,
				const real_t* __restrict__ position_z, const real_t* __restrict__ radius,
				const real_t* __restrict__ repulsion_coeff, const real_t* __restrict__ adhesion_coeff,
				const real_t* __restrict__ relative_maximum_adhesion_distance,
				const real_t* __restrict__ adhesion_affinity, const index_t* __restrict__ agent_type)
{
	for (index_t lhs = lhs_begin; lhs < lhs_end; ++lhs)
	{
		for (index_t rhs = 0; rhs < agents_count; ++rhs)
		{
			if (lhs == rhs)
				continue;

			real_t position_difference_x;
			real_t position_difference_y;
			real_t position_difference_z;

			position_difference_x = position_x[lhs] - position_x[rhs];
			if constexpr (dims > 1)
				position_difference_y = position_y[lhs] - position_y[rhs];
			if constexpr (dims > 2)
				position_difference_z = position_z[lhs] - position_z[rhs];

			real_t distance;
			if constexpr (dims == 1)
			{
				distance = std::abs(position_difference_x);
			}
			else if constexpr (dims == 2)
			{
				distance = std::sqrt(position_difference_x * position_difference_x
									 + position_difference_y * position_difference_y);
			}
			else // dims == 3
			{
				distance = std::sqrt(position_difference_x * position_difference_x
									 + position_difference_y * position_difference_y
									 + position_difference_z * position_difference_z);
			}

			distance = std::max<real_t>(distance, 0.00001);

			// compute repulsion
			real_t repulsion;
			{
				const real_t repulsive_distance = radius[lhs] + radius[rhs];

				repulsion = 1 - distance / repulsive_distance;

				repulsion = repulsion < 0 ? 0 : repulsion;

				repulsion *= repulsion;

				repulsion *= std::sqrt(repulsion_coeff[lhs] * repulsion_coeff[rhs]);
			}

			// compute adhesion
			real_t adhesion;
			{
				const real_t adhesion_distance = relative_maximum_adhesion_distance[lhs] * radius[lhs]
												 + relative_maximum_adhesion_distance[rhs] * radius[rhs];

				adhesion = 1 - distance / adhesion_distance;

				adhesion *= adhesion;

				const index_t lhs_type = agent_type[lhs];
				const index_t rhs_type = agent_type[rhs];

				adhesion *= std::sqrt(adhesion_coeff[lhs] * adhesion_coeff[rhs]
									  * adhesion_affinity[lhs * agent_types_count + rhs_type]
									  * adhesion_affinity[rhs * agent_types_count + lhs_type]);
			}

			real_t force = (repulsion - adhesion) / distance;

			velocity_x[lhs] += force * position_difference_x;
			if constexpr (dims > 1)
				velocity_y[lhs] += force * position_difference_y;
			if constexpr (dims > 2)
				velocity_z[lhs] += force * position_difference_z;
		}
	}
}

template <typename real_t>
void transposed_solver<real_t>::solve()
{
	static constexpr int block_size = 16;

#pragma omp parallel for schedule(static)
	for (index_t i = 0; i < agents_count_; i += block_size)
	{
		const auto lhs_begin = i;
		const auto lhs_end = std::min(i + block_size, agents_count_);

		if (dims_ == 1)
			solve_pair<1>(lhs_begin, lhs_end, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
		else if (dims_ == 2)
			solve_pair<2>(lhs_begin, lhs_end, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
		else if (dims_ == 3)
			solve_pair<3>(lhs_begin, lhs_end, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
	}

	// Update positions based on velocities
#pragma omp parallel for schedule(static)
	for (index_t i = 0; i < agents_count_; i++)
	{
		positionsx_[i] += velocitiesx_[i];
		if (dims_ > 1)
			positionsy_[i] += velocitiesy_[i];
		if (dims_ > 2)
			positionsz_[i] += velocitiesz_[i];
	}
}

template <typename real_t>
void transposed_solver<real_t>::initialize(const nlohmann::json&, const problem_t& problem)
{
	dims_ = static_cast<index_t>(problem.dims);
	agents_count_ = static_cast<index_t>(problem.agents_count);
	agent_types_count_ = static_cast<index_t>(problem.agent_types_count);

	// Initialize agent distributor
	agent_distributor<real_t> distributor(problem);

	// Access distributed agent data
	positionsx_ = std::make_unique<real_t[]>(agents_count_);
	positionsy_ = std::make_unique<real_t[]>(agents_count_);
	positionsz_ = std::make_unique<real_t[]>(agents_count_);

	velocitiesx_ = std::make_unique<real_t[]>(agents_count_);
	velocitiesy_ = std::make_unique<real_t[]>(agents_count_);
	velocitiesz_ = std::make_unique<real_t[]>(agents_count_);

	for (index_t i = 0; i < agents_count_; i++)
	{
		positionsx_[i] = distributor.positions_[i * dims_ + 0];
		velocitiesx_[i] = distributor.velocities_[i * dims_ + 0];
		if (dims_ > 1)
		{
			positionsy_[i] = distributor.positions_[i * dims_ + 1];
			velocitiesy_[i] = distributor.velocities_[i * dims_ + 1];
		}
		if (dims_ > 2)
		{
			positionsz_[i] = distributor.positions_[i * dims_ + 2];
			velocitiesz_[i] = distributor.velocities_[i * dims_ + 2];
		}
	}

	radius_ = std::move(distributor.radius_);
	repulsion_coeff_ = std::move(distributor.repulsion_coeff_);
	adhesion_coeff_ = std::move(distributor.adhesion_coeff_);
	max_adhesion_distance_ = std::move(distributor.max_adhesion_distance_);
	adhesion_affinity_ = std::move(distributor.adhesion_affinity_);
	agent_types_ = std::move(distributor.agent_types_);
}

template <typename real_t>
std::array<double, 3> transposed_solver<real_t>::access_agent(std::size_t agent_id)
{
	// Access agent data
	std::array<double, 3> agent_data = { 0.0, 0.0, 0.0 };
	agent_data[0] = static_cast<double>(positionsx_[agent_id]);
	if (dims_ > 1)
		agent_data[1] = static_cast<double>(positionsy_[agent_id]);
	if (dims_ > 2)
		agent_data[2] = static_cast<double>(positionsz_[agent_id]);
	return agent_data;
}

template <typename real_t>
void transposed_solver<real_t>::save(std::ostream& os) const
{
	// Save agent data to output stream
	for (std::size_t i = 0; i < static_cast<std::size_t>(agents_count_); i++)
	{
		os << "Agent " << i << ": ";
		os << positionsx_[i] << " ";
		os << positionsy_[i] << " ";
		os << positionsz_[i] << " ";
		os << std::endl;
	}
}

template class transposed_solver<float>;
template class transposed_solver<double>;
