#include "transposed_solver.h"

#include <cstdint>
#include <iostream>

#include <hwy/highway.h>

#include "../agent_distributor.h"

namespace hn = hwy::HWY_NAMESPACE;


template <std::size_t dims, typename index_t, typename real_t>
static constexpr void solve_pair_scalar(index_t lhs, index_t rhs, index_t agent_types_count,
										real_t* __restrict__ velocity_x, real_t* __restrict__ velocity_y,
										real_t* __restrict__ velocity_z, const real_t* __restrict__ position_x,
										const real_t* __restrict__ position_y, const real_t* __restrict__ position_z,
										const real_t* __restrict__ radius, const real_t* __restrict__ repulsion_coeff,
										const real_t* __restrict__ adhesion_coeff,
										const real_t* __restrict__ relative_maximum_adhesion_distance,
										const real_t* __restrict__ adhesion_affinity,
										const index_t* __restrict__ agent_type)
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

template <typename tag_t, typename simd_t>
static constexpr void slide_concat(simd_t& l, simd_t& r)
{
	l = hn::Slide1Down(tag_t(), l);
	auto first = hn::GetLane(r);
	l = hn::InsertLane(l, hn::Lanes(tag_t()) - 1, first);
	r = hn::Slide1Down(tag_t(), r);
}

template <std::size_t dims, typename index_t, typename real_t>
static constexpr void solve_pair(index_t lhs, index_t agents_count, index_t agent_types_count,
								 real_t* __restrict__ velocity_x, real_t* __restrict__ velocity_y,
								 real_t* __restrict__ velocity_z, const real_t* __restrict__ position_x,
								 const real_t* __restrict__ position_y, const real_t* __restrict__ position_z,
								 const real_t* __restrict__ radius, const real_t* __restrict__ repulsion_coeff,
								 const real_t* __restrict__ adhesion_coeff,
								 const real_t* __restrict__ relative_maximum_adhesion_distance,
								 const real_t* __restrict__ adhesion_affinity, const index_t* __restrict__ agent_type)
{
	using tag_t = hn::ScalableTag<real_t>;
	using simd_t = hn::Vec<tag_t>;
	using index_tag_t = hn::ScalableTag<index_t>;
	using index_simd_t = hn::Vec<index_tag_t>;
	HWY_LANES_CONSTEXPR index_t lanes = (index_t)hn::Lanes(tag_t());
	tag_t d;

	// Handle scalar remainder
	if (lhs + lanes > agents_count)
	{
		for (index_t i = lhs; i < agents_count; i++)
		{
			for (index_t j = 0; j < agents_count; j++)
			{
				// std::cout << "Solving pair: (" << i << ", " << j << ")" << std::endl;
				solve_pair_scalar<dims>(i, j, agent_types_count, velocity_x, velocity_y, velocity_z, position_x,
										position_y, position_z, radius, repulsion_coeff, adhesion_coeff,
										relative_maximum_adhesion_distance, adhesion_affinity, agent_type);
			}
		}

		return;
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

	simd_t rhs_radius_1 = hn::LoadU(tag_t(), radius);
	simd_t rhs_repulsion_coeff_1 = hn::LoadU(tag_t(), repulsion_coeff);
	simd_t rhs_adhesion_coeff_1 = hn::LoadU(tag_t(), adhesion_coeff);
	simd_t rhs_relative_maximum_adhesion_distance_1 = hn::LoadU(tag_t(), relative_maximum_adhesion_distance);
	index_simd_t rhs_agent_type_1 = hn::LoadU(index_tag_t(), agent_type);
	index_simd_t rhs_iota_1 = hn::Iota(index_tag_t(), 0);

	simd_t rhs_position_x_1 = hn::LoadU(tag_t(), position_x);
	simd_t rhs_position_y_1;
	simd_t rhs_position_z_1;

	if constexpr (dims > 1)
		rhs_position_y_1 = hn::LoadU(tag_t(), position_y);
	if constexpr (dims > 2)
		rhs_position_z_1 = hn::LoadU(tag_t(), position_z);

	for (index_t rhs = 0; rhs < agents_count;)
	{
		simd_t rhs_radius_2;
		simd_t rhs_repulsion_coeff_2;
		simd_t rhs_adhesion_coeff_2;
		simd_t rhs_relative_maximum_adhesion_distance_2;
		index_simd_t rhs_agent_type_2;
		index_simd_t rhs_iota_2;

		simd_t rhs_position_x_2;
		simd_t rhs_position_y_2;
		simd_t rhs_position_z_2;

		index_t remaining_shifts;

		{
			hn::MFromD<tag_t> mask;
			hn::MFromD<index_tag_t> index_mask;
			index_t rhs2;
			if (rhs + lanes < agents_count)
			{
				rhs2 = rhs + lanes;
				remaining_shifts = std::min(agents_count - (rhs + lanes), lanes);
				mask = hn::FirstN(tag_t(), remaining_shifts);
				index_mask = hn::FirstN(index_tag_t(), remaining_shifts);
			}
			else
			{
				rhs2 = 0;
				remaining_shifts = lanes;
				mask = hn::FirstN(tag_t(), lanes);
				index_mask = hn::FirstN(index_tag_t(), lanes);
			}

			rhs_radius_2 = hn::MaskedLoad(mask, tag_t(), radius + rhs2);
			rhs_repulsion_coeff_2 = hn::MaskedLoad(mask, tag_t(), repulsion_coeff + rhs2);
			rhs_adhesion_coeff_2 = hn::MaskedLoad(mask, tag_t(), adhesion_coeff + rhs2);
			rhs_relative_maximum_adhesion_distance_2 =
				hn::MaskedLoad(mask, tag_t(), relative_maximum_adhesion_distance + rhs2);
			rhs_agent_type_2 = hn::MaskedLoad(index_mask, index_tag_t(), agent_type + rhs2);
			rhs_iota_2 = hn::Iota(index_tag_t(), rhs2);

			rhs_position_x_2 = hn::MaskedLoad(mask, tag_t(), position_x + rhs2);
			if constexpr (dims > 1)
				rhs_position_y_2 = hn::MaskedLoad(mask, tag_t(), position_y + rhs2);
			if constexpr (dims > 2)
				rhs_position_z_2 = hn::MaskedLoad(mask, tag_t(), position_z + rhs2);
		}

		for (index_t lane_idx = 0; lane_idx < remaining_shifts; lane_idx++)
		{
			// std::cout << "Solving vector: ([" << lhs << "," << lhs + lanes << "], [" << (rhs + lane_idx) %
			// agents_count
			// 		  << "," << (rhs + lane_idx + lanes) % agents_count << "])" << std::endl;
			simd_t position_difference_x;
			simd_t position_difference_y;
			simd_t position_difference_z;

			position_difference_x = lhs_position_x - rhs_position_x_1;
			if constexpr (dims > 1)
				position_difference_y = lhs_position_y - rhs_position_y_1;
			if constexpr (dims > 2)
				position_difference_z = lhs_position_z - rhs_position_z_1;

			simd_t distance;
			if constexpr (dims == 1)
			{
				distance = hn::Abs(position_difference_x);
			}
			else if constexpr (dims == 2)
			{
				simd_t tmp = position_difference_x * position_difference_x;
				tmp = hn::MulAdd(position_difference_y, position_difference_y, tmp);
				distance = hn::Sqrt(tmp);
			}
			else // dims == 3
			{
				simd_t tmp = position_difference_x * position_difference_x;
				tmp = hn::MulAdd(position_difference_y, position_difference_y, tmp);
				tmp = hn::MulAdd(position_difference_z, position_difference_z, tmp);
				distance = hn::Sqrt(tmp);
			}

			distance = hn::Max(distance, hn::Set(d, 0.00001));

			// compute repulsion
			simd_t repulsion;
			{
				const simd_t repulsive_distance = lhs_radius + rhs_radius_1;

				repulsion = hn::Set(d, 1) - distance / repulsive_distance;

				repulsion = hn::Max(repulsion, hn::Set(d, 0));

				repulsion *= repulsion;

				repulsion *= hn::Sqrt(lhs_repulsion_coeff * rhs_repulsion_coeff_1);
			}

			// compute adhesion
			simd_t adhesion;
			{
				simd_t tmp = lhs_relative_maximum_adhesion_distance * lhs_radius;
				const simd_t adhesion_distance =
					hn::MulAdd(rhs_relative_maximum_adhesion_distance_1, rhs_radius_1, tmp);

				adhesion = hn::Set(d, 1) - distance / adhesion_distance;

				adhesion *= adhesion;

				index_simd_t lhs_index =
					hn::MulAdd(lhs_agent_type, hn::Set(index_tag_t(), agent_types_count), rhs_iota_1);
				index_simd_t rhs_index = hn::MulAdd(rhs_agent_type_1, hn::Set(index_tag_t(), agent_types_count),
													hn::Iota(index_tag_t(), lhs));

				simd_t lhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity, lhs_index);
				simd_t rhs_adhesion_affinity = hn::GatherIndex(tag_t(), adhesion_affinity, rhs_index);

				adhesion *=
					hn::Sqrt(lhs_adhesion_coeff * rhs_adhesion_coeff_1 * lhs_adhesion_affinity * rhs_adhesion_affinity);
			}

			simd_t force = (repulsion - adhesion) / distance;

			lhs_velocity_x = hn::MulAdd(force, position_difference_x, lhs_velocity_x);
			if constexpr (dims > 1)
				lhs_velocity_y = hn::MulAdd(force, position_difference_y, lhs_velocity_y);
			if constexpr (dims > 2)
				lhs_velocity_z = hn::MulAdd(force, position_difference_z, lhs_velocity_z);

			slide_concat<tag_t>(rhs_radius_1, rhs_radius_2);
			slide_concat<tag_t>(rhs_repulsion_coeff_1, rhs_repulsion_coeff_2);
			slide_concat<tag_t>(rhs_adhesion_coeff_1, rhs_adhesion_coeff_2);
			slide_concat<tag_t>(rhs_relative_maximum_adhesion_distance_1, rhs_relative_maximum_adhesion_distance_2);
			slide_concat<index_tag_t>(rhs_agent_type_1, rhs_agent_type_2);
			slide_concat<index_tag_t>(rhs_iota_1, rhs_iota_2);
			slide_concat<tag_t>(rhs_position_x_1, rhs_position_x_2);
			if constexpr (dims > 1)
				slide_concat<tag_t>(rhs_position_y_1, rhs_position_y_2);
			if constexpr (dims > 2)
				slide_concat<tag_t>(rhs_position_z_1, rhs_position_z_2);
		}

		rhs += remaining_shifts;
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
			solve_pair<1>(i, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
		else if (dims_ == 2)
			solve_pair<2>(i, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
						  velocitiesz_.get(), positionsx_.get(), positionsy_.get(), positionsz_.get(), radius_.get(),
						  repulsion_coeff_.get(), adhesion_coeff_.get(), max_adhesion_distance_.get(),
						  adhesion_affinity_.get(), agent_types_.get());
		else if (dims_ == 3)
			solve_pair<3>(i, agents_count_, agent_types_count_, velocitiesx_.get(), velocitiesy_.get(),
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
