#include "transposed_solver.h"

#include <cstdint>
#include <iostream>

#include <hwy/highway.h>

#include "../agent_distributor.h"

namespace hn = hwy::HWY_NAMESPACE;

template <std::size_t dims, typename index_t, typename real_t>
static constexpr void solve_pair_scalar(
	bool try_skip_repulsion, bool try_skip_adhesion, index_t lhs, index_t rhs, index_t agent_types_count,
	real_t* __restrict__ velocity_x, real_t* __restrict__ velocity_y, real_t* __restrict__ velocity_z,
	const real_t* __restrict__ position_x, const real_t* __restrict__ position_y, const real_t* __restrict__ position_z,
	const real_t* __restrict__ radius, const real_t* __restrict__ repulsion_coeff,
	const real_t* __restrict__ adhesion_coeff, const real_t* __restrict__ relative_maximum_adhesion_distance,
	const real_t* __restrict__ adhesion_affinity, const index_t* __restrict__ agent_type)
{
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
		distance =
			std::sqrt(position_difference_x * position_difference_x + position_difference_y * position_difference_y);
	}
	else // dims == 3
	{
		distance =
			std::sqrt(position_difference_x * position_difference_x + position_difference_y * position_difference_y
					  + position_difference_z * position_difference_z);
	}

	distance = std::max<real_t>(distance, 0.00001);

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

	velocity_x[lhs] += force * position_difference_x;
	if constexpr (dims > 1)
		velocity_y[lhs] += force * position_difference_y;
	if constexpr (dims > 2)
		velocity_z[lhs] += force * position_difference_z;
}

template <std::size_t dims, typename index_t, typename real_t>
static constexpr void solve_pair(bool try_skip_repulsion, bool try_skip_adhesion, index_t lhs, index_t agents_count,
								 index_t agent_types_count, real_t* __restrict__ velocity_x,
								 real_t* __restrict__ velocity_y, real_t* __restrict__ velocity_z,
								 const real_t* __restrict__ position_x, const real_t* __restrict__ position_y,
								 const real_t* __restrict__ position_z, const real_t* __restrict__ radius,
								 const real_t* __restrict__ repulsion_coeff, const real_t* __restrict__ adhesion_coeff,
								 const real_t* __restrict__ relative_maximum_adhesion_distance,
								 const real_t* __restrict__ adhesion_affinity, const index_t* __restrict__ agent_type)
{
	using tag_t = hn::ScalableTag<real_t>;
	using simd_t = hn::Vec<tag_t>;
	using index_tag_t = hn::ScalableTag<index_t>;
	using index_simd_t = hn::Vec<index_tag_t>;
	HWY_LANES_CONSTEXPR index_t lanes = (index_t)hn::Lanes(tag_t());

	// Handle scalar remainder
	if (lhs + lanes > agents_count)
	{
		for (index_t i = lhs; i < agents_count; i++)
		{
			for (index_t j = 0; j < agents_count; j++)
			{
				if (i == j)
					continue;

				// std::cout << "Solving pair: (" << i << ", " << j << ")" << std::endl;
				solve_pair_scalar<dims>(try_skip_repulsion, try_skip_adhesion, i, j, agent_types_count, velocity_x,
										velocity_y, velocity_z, position_x, position_y, position_z, radius,
										repulsion_coeff, adhesion_coeff, relative_maximum_adhesion_distance,
										adhesion_affinity, agent_type);
			}
		}

		return;
	}

	// Handle scalar triangle begin
	for (index_t i = 1; i < lanes; i++)
	{
		for (index_t j = 0; j < i; j++)
		{
			// std::cout << "Solving pair: (" << (lhs + i) << ", " << j << ")" << std::endl;
			solve_pair_scalar<dims>(try_skip_repulsion, try_skip_adhesion, lhs + i, j, agent_types_count, velocity_x,
									velocity_y, velocity_z, position_x, position_y, position_z, radius, repulsion_coeff,
									adhesion_coeff, relative_maximum_adhesion_distance, adhesion_affinity, agent_type);
		}
	}

	// Handle scalar triangle end
	for (index_t i = 0; i < lanes - 1; i++)
	{
		for (index_t j = agents_count - lanes + i + 1; j < agents_count; j++)
		{
			// std::cout << "Solving pair: (" << (lhs + i) << ", " << j << ")" << std::endl;
			solve_pair_scalar<dims>(try_skip_repulsion, try_skip_adhesion, lhs + i, j, agent_types_count, velocity_x,
									velocity_y, velocity_z, position_x, position_y, position_z, radius, repulsion_coeff,
									adhesion_coeff, relative_maximum_adhesion_distance, adhesion_affinity, agent_type);
		}
	}

	const simd_t lhs_radius = hn::LoadU(tag_t(), radius + lhs);
	const simd_t lhs_repulsion_coeff = hn::LoadU(tag_t(), repulsion_coeff + lhs);
	const simd_t lhs_adhesion_coeff = hn::LoadU(tag_t(), adhesion_coeff + lhs);
	const simd_t lhs_relative_maximum_adhesion_distance = hn::LoadU(tag_t(), relative_maximum_adhesion_distance + lhs);
	const index_simd_t lhs_agent_type = hn::LoadU(index_tag_t(), agent_type + lhs);

	const simd_t lhs_position_x = hn::LoadU(tag_t(), position_x + lhs);
	simd_t lhs_position_y;
	simd_t lhs_position_z;

	if constexpr (dims > 1)
		lhs_position_y = hn::LoadU(tag_t(), position_y + lhs);
	if constexpr (dims > 2)
		lhs_position_z = hn::LoadU(tag_t(), position_z + lhs);

	simd_t lhs_velocity_x = hn::LoadU(tag_t(), velocity_x + lhs);
	simd_t lhs_velocity_y;
	simd_t lhs_velocity_z;
	if constexpr (dims > 1)
		lhs_velocity_y = hn::LoadU(tag_t(), velocity_y + lhs);
	if constexpr (dims > 2)
		lhs_velocity_z = hn::LoadU(tag_t(), velocity_z + lhs);

	for (index_t rhs = 0; rhs < agents_count - lanes + 1; rhs++)
	{
		if (rhs == lhs)
			continue;
		// std::cout << "Solving vector: ([" << lhs << "," << lhs + lanes << "], [" << rhs << "," << rhs + lanes << "])"
		// 		  << std::endl;

		const simd_t rhs_radius = hn::LoadU(tag_t(), radius + rhs);
		const simd_t rhs_repulsion_coeff = hn::LoadU(tag_t(), repulsion_coeff + rhs);
		const simd_t rhs_adhesion_coeff = hn::LoadU(tag_t(), adhesion_coeff + rhs);
		const simd_t rhs_relative_maximum_adhesion_distance =
			hn::LoadU(tag_t(), relative_maximum_adhesion_distance + rhs);
		const index_simd_t rhs_agent_type = hn::LoadU(index_tag_t(), agent_type + rhs);

		const simd_t rhs_position_x = hn::LoadU(tag_t(), position_x + rhs);
		simd_t rhs_position_y;
		simd_t rhs_position_z;

		if constexpr (dims > 1)
			rhs_position_y = hn::LoadU(tag_t(), position_y + rhs);
		if constexpr (dims > 2)
			rhs_position_z = hn::LoadU(tag_t(), position_z + rhs);

		{
			simd_t position_difference_x;
			simd_t position_difference_y;
			simd_t position_difference_z;

			position_difference_x = hn::Sub(lhs_position_x, rhs_position_x);
			if constexpr (dims > 1)
				position_difference_y = hn::Sub(lhs_position_y, rhs_position_y);
			if constexpr (dims > 2)
				position_difference_z = hn::Sub(lhs_position_z, rhs_position_z);

			simd_t distance;
			if constexpr (dims == 1)
			{
				distance = hn::Abs(position_difference_x);
			}
			else if constexpr (dims == 2)
			{
				simd_t tmp = hn::Mul(position_difference_x, position_difference_x);
				tmp = hn::MulAdd(position_difference_y, position_difference_y, tmp);
				distance = hn::Sqrt(tmp);
			}
			else // dims == 3
			{
				simd_t tmp = hn::Mul(position_difference_x, position_difference_x);
				tmp = hn::MulAdd(position_difference_y, position_difference_y, tmp);
				tmp = hn::MulAdd(position_difference_z, position_difference_z, tmp);
				distance = hn::Sqrt(tmp);
			}

			distance = hn::Max(distance, hn::Set(tag_t(), 0.00001));

			// compute repulsion
			simd_t repulsion;
			if (try_skip_repulsion)
			{
				repulsion = hn::Zero(tag_t());
				const simd_t repulsive_distance = hn::Add(lhs_radius, rhs_radius);

				if (!hn::AllTrue(tag_t(), hn::Gt(distance, repulsive_distance)))
				{
					repulsion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, repulsive_distance));
					repulsion *= repulsion;
					repulsion *= hn::Sqrt(lhs_repulsion_coeff * rhs_repulsion_coeff);
				}
			}
			else
			{
				const simd_t repulsive_distance = hn::Add(lhs_radius, rhs_radius);

				repulsion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, repulsive_distance));
				repulsion = hn::Max(repulsion, hn::Zero(tag_t()));
				repulsion *= repulsion;
				repulsion *= hn::Sqrt(lhs_repulsion_coeff * rhs_repulsion_coeff);
			}

			// compute adhesion
			simd_t adhesion;
			if (try_skip_adhesion)
			{
				adhesion = hn::Zero(tag_t());
				simd_t tmp = hn::Mul(lhs_relative_maximum_adhesion_distance, lhs_radius);
				const simd_t adhesion_distance = hn::MulAdd(rhs_relative_maximum_adhesion_distance, rhs_radius, tmp);

				if (!hn::AllTrue(tag_t(), hn::Gt(distance, adhesion_distance)))
				{
					adhesion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, adhesion_distance));

					adhesion *= adhesion;

					const index_simd_t lhs_indices = hn::Iota(index_tag_t(), lhs);
					const index_simd_t rhs_indices = hn::Iota(index_tag_t(), rhs);
					const index_simd_t types_count = hn::Set(index_tag_t(), agent_types_count);

					index_simd_t lhs_index = hn::MulAdd(lhs_indices, types_count, rhs_agent_type);
					index_simd_t rhs_index = hn::MulAdd(rhs_indices, types_count, lhs_agent_type);

					simd_t lhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity, lhs_index);
					simd_t rhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity, rhs_index);

					adhesion *= hn::Sqrt(hn::Mul(hn::Mul(lhs_adhesion_coeff, rhs_adhesion_coeff),
												 hn::Mul(lhs_adhesion_affinity, rhs_adhesion_affinity)));
				}
			}
			else
			{
				simd_t tmp = hn::Mul(lhs_relative_maximum_adhesion_distance, lhs_radius);
				const simd_t adhesion_distance = hn::MulAdd(rhs_relative_maximum_adhesion_distance, rhs_radius, tmp);

				adhesion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, adhesion_distance));
				adhesion = hn::Max(adhesion, hn::Zero(tag_t()));
				adhesion *= adhesion;

				const index_simd_t lhs_indices = hn::Iota(index_tag_t(), lhs);
				const index_simd_t rhs_indices = hn::Iota(index_tag_t(), rhs);
				const index_simd_t types_count = hn::Set(index_tag_t(), agent_types_count);

				index_simd_t lhs_index = hn::MulAdd(lhs_indices, types_count, rhs_agent_type);
				index_simd_t rhs_index = hn::MulAdd(rhs_indices, types_count, lhs_agent_type);

				simd_t lhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity, lhs_index);
				simd_t rhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity, rhs_index);

				adhesion *= hn::Sqrt(hn::Mul(hn::Mul(lhs_adhesion_coeff, rhs_adhesion_coeff),
											 hn::Mul(lhs_adhesion_affinity, rhs_adhesion_affinity)));
			}

			simd_t force = hn::Div(hn::Sub(repulsion, adhesion), distance);

			lhs_velocity_x = hn::MulAdd(force, position_difference_x, lhs_velocity_x);
			if constexpr (dims > 1)
				lhs_velocity_y = hn::MulAdd(force, position_difference_y, lhs_velocity_y);
			if constexpr (dims > 2)
				lhs_velocity_z = hn::MulAdd(force, position_difference_z, lhs_velocity_z);
		}
	}

	hn::StoreU(lhs_velocity_x, tag_t(), velocity_x + lhs);
	if constexpr (dims > 1)
		hn::StoreU(lhs_velocity_y, tag_t(), velocity_y + lhs);
	if constexpr (dims > 2)
		hn::StoreU(lhs_velocity_z, tag_t(), velocity_z + lhs);
}

template <typename real_t>
void transposed_solver<real_t>::solve()
{
	using tag_t = hn::ScalableTag<real_t>;
	HWY_LANES_CONSTEXPR int block_size = hn::Lanes(tag_t());

#pragma omp parallel for schedule(static)
	for (index_t i = 0; i < agents_count_; i += block_size)
	{
		if (dims_ == 1)
			solve_pair<1>(try_skip_repulsion_, try_skip_adhesion_, i, agents_count_, agent_types_count_,
						  velocitiesx_.get(), velocitiesy_.get(), velocitiesz_.get(), positionsx_.get(),
						  positionsy_.get(), positionsz_.get(), radius_.get(), repulsion_coeff_.get(),
						  adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
						  agent_types_.get());
		else if (dims_ == 2)
			solve_pair<2>(try_skip_repulsion_, try_skip_adhesion_, i, agents_count_, agent_types_count_,
						  velocitiesx_.get(), velocitiesy_.get(), velocitiesz_.get(), positionsx_.get(),
						  positionsy_.get(), positionsz_.get(), radius_.get(), repulsion_coeff_.get(),
						  adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
						  agent_types_.get());
		else if (dims_ == 3)
			solve_pair<3>(try_skip_repulsion_, try_skip_adhesion_, i, agents_count_, agent_types_count_,
						  velocitiesx_.get(), velocitiesy_.get(), velocitiesz_.get(), positionsx_.get(),
						  positionsy_.get(), positionsz_.get(), radius_.get(), repulsion_coeff_.get(),
						  adhesion_coeff_.get(), max_adhesion_distance_.get(), adhesion_affinity_.get(),
						  agent_types_.get());
	}

// Update positions based on velocities
#pragma omp parallel for schedule(static)
	for (index_t i = 0; i < agents_count_; i++)
	{
		positionsx_[i] += velocitiesx_[i] * timestep_;
		velocitiesx_[i] = 0;
		if (dims_ > 1)
		{
			positionsy_[i] += velocitiesy_[i] * timestep_;
			velocitiesy_[i] = 0;
		}
		if (dims_ > 2)
		{
			positionsz_[i] += velocitiesz_[i] * timestep_;
			velocitiesz_[i] = 0;
		}
	}
}

template <typename real_t>
void transposed_solver<real_t>::initialize(const nlohmann::json& params, const problem_t& problem)
{
	try_skip_repulsion_ = params.value("try_skip_repulsion", false);
	try_skip_adhesion_ = params.value("try_skip_adhesion", false);

	dims_ = static_cast<index_t>(problem.dims);
	timestep_ = static_cast<real_t>(problem.timestep);
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
