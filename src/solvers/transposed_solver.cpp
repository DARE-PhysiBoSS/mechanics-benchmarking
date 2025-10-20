#include "transposed_solver.h"

#include <cstdint>
#include <iostream>

#include <hwy/highway.h>

#include "../agent_distributor.h"

namespace hn = hwy::HWY_NAMESPACE;

template <std::size_t dims, typename index_t, typename real_t>
void solve_pair(index_t lhs_begin, index_t agents_count, index_t agent_types_count, real_t* __restrict__ velocity_x,
				real_t* __restrict__ velocity_y, real_t* __restrict__ velocity_z, const real_t* __restrict__ position_x,
				const real_t* __restrict__ position_y, const real_t* __restrict__ position_z,
				const real_t* __restrict__ radius, const real_t* __restrict__ repulsion_coeff,
				const real_t* __restrict__ adhesion_coeff,
				const real_t* __restrict__ relative_maximum_adhesion_distance,
				const real_t* __restrict__ adhesion_affinity, const index_t* __restrict__ agent_type)
{
	using tag_t = hn::ScalableTag<real_t>;
	tag_t d;
	using simd_t = hn::Vec<tag_t>;

	using index_tag_t = hn::FixedTag<index_t, hn::MaxLanes(tag_t())>;
	using index_t_simd_t = hn::Vec<index_tag_t>;

	simd_t lhs_radius = hn::LoadU(tag_t(), radius + lhs_begin);
	simd_t lhs_repulsion_coeff = hn::LoadU(tag_t(), repulsion_coeff + lhs_begin);
	simd_t lhs_adhesion_coeff = hn::LoadU(tag_t(), adhesion_coeff + lhs_begin);
	simd_t lhs_relative_maximum_adhesion_distance = hn::LoadU(tag_t(), relative_maximum_adhesion_distance + lhs_begin);
	index_t_simd_t lhs_agent_type = hn::LoadU(index_tag_t(), agent_type + lhs_begin);

	simd_t lhs_position_x = hn::LoadU(tag_t(), position_x + lhs_begin);
	simd_t lhs_position_y;
	simd_t lhs_position_z;

	if constexpr (dims > 1)
		lhs_position_y = hn::LoadU(tag_t(), position_y + lhs_begin);
	if constexpr (dims > 2)
		lhs_position_z = hn::LoadU(tag_t(), position_z + lhs_begin);

	simd_t lhs_velocity_x = hn::LoadU(tag_t(), velocity_x + lhs_begin);
	simd_t lhs_velocity_y;
	simd_t lhs_velocity_z;
	if constexpr (dims > 1)
		lhs_velocity_y = hn::LoadU(tag_t(), velocity_y + lhs_begin);
	if constexpr (dims > 2)
		lhs_velocity_z = hn::LoadU(tag_t(), velocity_z + lhs_begin);

	for (index_t rhs = 0; rhs < agents_count; rhs += hn::Lanes(tag_t()))
	{
		simd_t position_difference_x;
		simd_t position_difference_y;
		simd_t position_difference_z;

		position_difference_x = lhs_position_x - hn::LoadU(tag_t(), position_x + rhs);
		if constexpr (dims > 1)
			position_difference_y = lhs_position_y - hn::LoadU(tag_t(), position_y + rhs);
		if constexpr (dims > 2)
			position_difference_z = lhs_position_z - hn::LoadU(tag_t(), position_z + rhs);

		simd_t distance;
		if constexpr (dims == 1)
		{
			distance = hn::Abs(position_difference_x);
		}
		else if constexpr (dims == 2)
		{
			distance =
				hn::Sqrt(position_difference_x * position_difference_x + position_difference_y * position_difference_y);
		}
		else // dims == 3
		{
			distance =
				hn::Sqrt(position_difference_x * position_difference_x + position_difference_y * position_difference_y
						 + position_difference_z * position_difference_z);
		}

		distance = hn::Max(distance, hn::Set(d, 0.00001));

		// compute repulsion
		simd_t repulsion;
		{
			const simd_t repulsive_distance = lhs_radius + hn::LoadU(tag_t(), radius + rhs);

			repulsion = hn::Set(d, 1) - distance / repulsive_distance;

			repulsion = hn::Max(repulsion, hn::Set(d, 0));

			repulsion *= repulsion;

			repulsion *= hn::Sqrt(lhs_repulsion_coeff * hn::LoadU(tag_t(), repulsion_coeff + rhs));
		}

		// compute adhesion
		simd_t adhesion;
		{
			const simd_t adhesion_distance =
				lhs_relative_maximum_adhesion_distance * lhs_radius
				+ hn::LoadU(tag_t(), relative_maximum_adhesion_distance + rhs) * hn::LoadU(tag_t(), radius + rhs);

			adhesion = hn::Set(d, 1) - distance / adhesion_distance;

			adhesion *= adhesion;

			simd_t rhs_adhesion_affinity =
				hn::GatherIndex(tag_t(), adhesion_affinity + rhs * agent_types_count, lhs_agent_type);
			simd_t lhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity + lhs_begin * agent_types_count,
														   hn::LoadU(index_tag_t(), agent_type + rhs));

			adhesion *= hn::Sqrt(lhs_adhesion_coeff * hn::LoadU(tag_t(), adhesion_coeff + rhs) * lhs_adhesion_affinity
								 * rhs_adhesion_affinity);
		}

		simd_t force = (repulsion - adhesion) / distance;

		lhs_velocity_x += force * position_difference_x;
		if constexpr (dims > 1)
			lhs_velocity_y += force * position_difference_y;
		if constexpr (dims > 2)
			lhs_velocity_z += force * position_difference_z;
	}
}

template <typename real_t>
void transposed_solver<real_t>::solve()
{
	using tag_t = hn::ScalableTag<real_t>;
	HWY_LANES_CONSTEXPR int block_size = hn::Lanes(tag_t());
	std::cout << "Block size: " << block_size << std::endl;

	// #pragma omp parallel for schedule(static)
	for (index_t i = 0; i < agents_count_; i += block_size)
	{
		const auto lhs_begin = i;

		if (dims_ == 1)
			solve_pair<1>(lhs_begin, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
		else if (dims_ == 2)
			solve_pair<2>(lhs_begin, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
		else if (dims_ == 3)
			solve_pair<3>(lhs_begin, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
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

	if constexpr (std::is_same_v<int32_t, index_t>)
		agent_types_ = std::move(distributor.agent_types_);
	else
	{
		agent_types_ = std::make_unique<index_t[]>(agents_count_);
		for (index_t i = 0; i < agents_count_; i++)
		{
			agent_types_[i] = distributor.agent_types_[i];
		}
	}
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
