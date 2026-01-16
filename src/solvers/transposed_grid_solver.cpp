#include "transposed_grid_solver.h"

#include <cstdint>
#include <iostream>

#include <hwy/highway.h>

#include "../agent_distributor.h"

namespace hn = hwy::HWY_NAMESPACE;

template <std::size_t dims, typename index_t, typename real_t>
static constexpr void solve_pair_scalar(
	bool try_skip_repulsion, bool try_skip_adhesion, index_t lhs, index_t rhs, index_t agent_types_count,
	real_t* __restrict__ lhs_velocity_x, real_t* __restrict__ lhs_velocity_y, real_t* __restrict__ lhs_velocity_z,
	const real_t* __restrict__ lhs_position_x, const real_t* __restrict__ lhs_position_y,
	const real_t* __restrict__ lhs_position_z, const real_t* __restrict__ lhs_radius,
	const real_t* __restrict__ lhs_repulsion_coeff, const real_t* __restrict__ lhs_adhesion_coeff,
	const real_t* __restrict__ lhs_relative_maximum_adhesion_distance, const real_t* __restrict__ lhs_adhesion_affinity,
	const index_t* __restrict__ lhs_agent_type, const real_t* __restrict__ rhs_position_x,
	const real_t* __restrict__ rhs_position_y, const real_t* __restrict__ rhs_position_z,
	const real_t* __restrict__ rhs_radius, const real_t* __restrict__ rhs_repulsion_coeff,
	const real_t* __restrict__ rhs_adhesion_coeff, const real_t* __restrict__ rhs_relative_maximum_adhesion_distance,
	const real_t* __restrict__ rhs_adhesion_affinity, const index_t* __restrict__ rhs_agent_type)
{
	real_t position_difference_x;
	real_t position_difference_y;
	real_t position_difference_z;

	position_difference_x = lhs_position_x[lhs] - rhs_position_x[rhs];
	if constexpr (dims > 1)
		position_difference_y = lhs_position_y[lhs] - rhs_position_y[rhs];
	if constexpr (dims > 2)
		position_difference_z = lhs_position_z[lhs] - rhs_position_z[rhs];

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
		const real_t repulsive_distance = lhs_radius[lhs] + rhs_radius[rhs];

		if (distance < repulsive_distance)
		{
			repulsion = 1 - distance / repulsive_distance;

			repulsion *= repulsion;

			repulsion *= std::sqrt(lhs_repulsion_coeff[lhs] * rhs_repulsion_coeff[rhs]);
		}
	}
	else
	{
		const real_t repulsive_distance = lhs_radius[lhs] + rhs_radius[rhs];

		repulsion = 1 - distance / repulsive_distance;

		repulsion = repulsion < 0 ? 0 : repulsion;

		repulsion *= repulsion;

		repulsion *= std::sqrt(lhs_repulsion_coeff[lhs] * rhs_repulsion_coeff[rhs]);
	}

	// compute adhesion
	real_t adhesion;
	if (try_skip_adhesion)
	{
		adhesion = 0;
		const real_t adhesion_distance = lhs_relative_maximum_adhesion_distance[lhs] * lhs_radius[lhs]
										 + rhs_relative_maximum_adhesion_distance[rhs] * rhs_radius[rhs];

		if (distance < adhesion_distance)
		{
			adhesion = 1 - distance / adhesion_distance;

			adhesion *= adhesion;

			const index_t lhs_type = lhs_agent_type[lhs];
			const index_t rhs_type = rhs_agent_type[rhs];

			adhesion *= std::sqrt(lhs_adhesion_coeff[lhs] * rhs_adhesion_coeff[rhs]
								  * lhs_adhesion_affinity[lhs * agent_types_count + rhs_type]
								  * rhs_adhesion_affinity[rhs * agent_types_count + lhs_type]);
		}
	}
	else
	{
		const real_t adhesion_distance = lhs_relative_maximum_adhesion_distance[lhs] * lhs_radius[lhs]
										 + rhs_relative_maximum_adhesion_distance[rhs] * rhs_radius[rhs];

		adhesion = 1 - distance / adhesion_distance;

		adhesion = adhesion < 0 ? 0 : adhesion;

		adhesion *= adhesion;

		const index_t lhs_type = lhs_agent_type[lhs];
		const index_t rhs_type = rhs_agent_type[rhs];

		adhesion *= std::sqrt(lhs_adhesion_coeff[lhs] * rhs_adhesion_coeff[rhs]
							  * lhs_adhesion_affinity[lhs * agent_types_count + rhs_type]
							  * rhs_adhesion_affinity[rhs * agent_types_count + lhs_type]);
	}

	real_t force = (repulsion - adhesion) / distance;

	lhs_velocity_x[lhs] += force * position_difference_x;
	if constexpr (dims > 1)
		lhs_velocity_y[lhs] += force * position_difference_y;
	if constexpr (dims > 2)
		lhs_velocity_z[lhs] += force * position_difference_z;
}

template <std::size_t dims, typename index_t, typename real_t>
static constexpr void solve_pair(
	bool try_skip_repulsion, bool try_skip_adhesion, index_t lhs_count, index_t rhs_count, index_t agent_types_count,
	real_t* __restrict__ lhs_velocity_x, real_t* __restrict__ lhs_velocity_y, real_t* __restrict__ lhs_velocity_z,
	const real_t* __restrict__ lhs_position_x, const real_t* __restrict__ lhs_position_y,
	const real_t* __restrict__ lhs_position_z, const real_t* __restrict__ lhs_radius,
	const real_t* __restrict__ lhs_repulsion_coeff, const real_t* __restrict__ lhs_adhesion_coeff,
	const real_t* __restrict__ lhs_relative_maximum_adhesion_distance, const real_t* __restrict__ lhs_adhesion_affinity,
	const index_t* __restrict__ lhs_agent_type, const real_t* __restrict__ rhs_position_x,
	const real_t* __restrict__ rhs_position_y, const real_t* __restrict__ rhs_position_z,
	const real_t* __restrict__ rhs_radius, const real_t* __restrict__ rhs_repulsion_coeff,
	const real_t* __restrict__ rhs_adhesion_coeff, const real_t* __restrict__ rhs_relative_maximum_adhesion_distance,
	const real_t* __restrict__ rhs_adhesion_affinity, const index_t* __restrict__ rhs_agent_type)
{
	using tag_t = hn::ScalableTag<real_t>;
	using simd_t = hn::Vec<tag_t>;
	using index_tag_t = hn::ScalableTag<index_t>;
	using index_simd_t = hn::Vec<index_tag_t>;
	HWY_LANES_CONSTEXPR index_t lanes = (index_t)hn::Lanes(tag_t());

	if (rhs_count < lanes)
	{
		// Process all lhs agents against all rhs agents (scalar)
		for (index_t lhs = 0; lhs < lhs_count; lhs++)
		{
			for (index_t rhs = 0; rhs < rhs_count; rhs++)
			{
				// std::cout << "Solving pair: (" << lhs << ", " << rhs << ")" << std::endl;
				solve_pair_scalar<dims>(
					try_skip_repulsion, try_skip_adhesion, lhs, rhs, agent_types_count, lhs_velocity_x, lhs_velocity_y,
					lhs_velocity_z, lhs_position_x, lhs_position_y, lhs_position_z, lhs_radius, lhs_repulsion_coeff,
					lhs_adhesion_coeff, lhs_relative_maximum_adhesion_distance, lhs_adhesion_affinity, lhs_agent_type,
					rhs_position_x, rhs_position_y, rhs_position_z, rhs_radius, rhs_repulsion_coeff, rhs_adhesion_coeff,
					rhs_relative_maximum_adhesion_distance, rhs_adhesion_affinity, rhs_agent_type);
			}
		}

		return;
	}

	// Process all lhs agents against all rhs agents
	for (index_t lhs = 0; lhs < lhs_count; lhs += lanes)
	{
		// Handle scalar remainder
		if (lhs + lanes > lhs_count)
		{
			for (index_t i = lhs; i < lhs_count; i++)
			{
				for (index_t j = 0; j < rhs_count; j++)
				{
					// std::cout << "Solving pair: (" << i << ", " << j << ")" << std::endl;
					solve_pair_scalar<dims>(
						try_skip_repulsion, try_skip_adhesion, i, j, agent_types_count, lhs_velocity_x, lhs_velocity_y,
						lhs_velocity_z, lhs_position_x, lhs_position_y, lhs_position_z, lhs_radius, lhs_repulsion_coeff,
						lhs_adhesion_coeff, lhs_relative_maximum_adhesion_distance, lhs_adhesion_affinity,
						lhs_agent_type, rhs_position_x, rhs_position_y, rhs_position_z, rhs_radius, rhs_repulsion_coeff,
						rhs_adhesion_coeff, rhs_relative_maximum_adhesion_distance, rhs_adhesion_affinity,
						rhs_agent_type);
				}
			}

			continue;
		}

		// Handle scalar triangle begin
		for (index_t i = 1; i < lanes; i++)
		{
			for (index_t j = 0; j < i; j++)
			{
				// std::cout << "Solving pair: (" << (lhs + i) << ", " << j << ")" << std::endl;
				solve_pair_scalar<dims>(try_skip_repulsion, try_skip_adhesion, lhs + i, j, agent_types_count,
										lhs_velocity_x, lhs_velocity_y, lhs_velocity_z, lhs_position_x, lhs_position_y,
										lhs_position_z, lhs_radius, lhs_repulsion_coeff, lhs_adhesion_coeff,
										lhs_relative_maximum_adhesion_distance, lhs_adhesion_affinity, lhs_agent_type,
										rhs_position_x, rhs_position_y, rhs_position_z, rhs_radius, rhs_repulsion_coeff,
										rhs_adhesion_coeff, rhs_relative_maximum_adhesion_distance,
										rhs_adhesion_affinity, rhs_agent_type);
			}
		}

		// Handle scalar triangle end
		for (index_t i = 0; i < lanes - 1; i++)
		{
			for (index_t j = rhs_count - lanes + i + 1; j < rhs_count; j++)
			{
				// std::cout << "Solving pair: (" << (lhs + i) << ", " << j << ")" << std::endl;
				solve_pair_scalar<dims>(try_skip_repulsion, try_skip_adhesion, lhs + i, j, agent_types_count,
										lhs_velocity_x, lhs_velocity_y, lhs_velocity_z, lhs_position_x, lhs_position_y,
										lhs_position_z, lhs_radius, lhs_repulsion_coeff, lhs_adhesion_coeff,
										lhs_relative_maximum_adhesion_distance, lhs_adhesion_affinity, lhs_agent_type,
										rhs_position_x, rhs_position_y, rhs_position_z, rhs_radius, rhs_repulsion_coeff,
										rhs_adhesion_coeff, rhs_relative_maximum_adhesion_distance,
										rhs_adhesion_affinity, rhs_agent_type);
			}
		}

		const simd_t lhs_radius_vec = hn::LoadU(tag_t(), lhs_radius + lhs);
		const simd_t lhs_repulsion_coeff_vec = hn::LoadU(tag_t(), lhs_repulsion_coeff + lhs);
		const simd_t lhs_adhesion_coeff_vec = hn::LoadU(tag_t(), lhs_adhesion_coeff + lhs);
		const simd_t lhs_relative_maximum_adhesion_distance_vec =
			hn::LoadU(tag_t(), lhs_relative_maximum_adhesion_distance + lhs);
		const index_simd_t lhs_agent_type_vec = hn::LoadU(index_tag_t(), lhs_agent_type + lhs);

		const simd_t lhs_position_x_vec = hn::LoadU(tag_t(), lhs_position_x + lhs);
		simd_t lhs_position_y_vec;
		simd_t lhs_position_z_vec;

		if constexpr (dims > 1)
			lhs_position_y_vec = hn::LoadU(tag_t(), lhs_position_y + lhs);
		if constexpr (dims > 2)
			lhs_position_z_vec = hn::LoadU(tag_t(), lhs_position_z + lhs);

		simd_t lhs_velocity_x_vec = hn::LoadU(tag_t(), lhs_velocity_x + lhs);
		simd_t lhs_velocity_y_vec;
		simd_t lhs_velocity_z_vec;
		if constexpr (dims > 1)
			lhs_velocity_y_vec = hn::LoadU(tag_t(), lhs_velocity_y + lhs);
		if constexpr (dims > 2)
			lhs_velocity_z_vec = hn::LoadU(tag_t(), lhs_velocity_z + lhs);

		for (index_t rhs = 0; rhs < rhs_count - lanes + 1; rhs++)
		{
			// std::cout << "Solving vector: ([" << lhs << "," << lhs + lanes << "], [" << rhs << "," << rhs + lanes
			// << "])"
			// 		  << std::endl;

			const simd_t rhs_radius_vec = hn::LoadU(tag_t(), rhs_radius + rhs);
			const simd_t rhs_repulsion_coeff_vec = hn::LoadU(tag_t(), rhs_repulsion_coeff + rhs);
			const simd_t rhs_adhesion_coeff_vec = hn::LoadU(tag_t(), rhs_adhesion_coeff + rhs);
			const simd_t rhs_relative_maximum_adhesion_distance_vec =
				hn::LoadU(tag_t(), rhs_relative_maximum_adhesion_distance + rhs);
			const index_simd_t rhs_agent_type_vec = hn::LoadU(index_tag_t(), rhs_agent_type + rhs);

			const simd_t rhs_position_x_vec = hn::LoadU(tag_t(), rhs_position_x + rhs);
			simd_t rhs_position_y_vec;
			simd_t rhs_position_z_vec;

			if constexpr (dims > 1)
				rhs_position_y_vec = hn::LoadU(tag_t(), rhs_position_y + rhs);
			if constexpr (dims > 2)
				rhs_position_z_vec = hn::LoadU(tag_t(), rhs_position_z + rhs);

			{
				simd_t position_difference_x;
				simd_t position_difference_y;
				simd_t position_difference_z;

				position_difference_x = hn::Sub(lhs_position_x_vec, rhs_position_x_vec);
				if constexpr (dims > 1)
					position_difference_y = hn::Sub(lhs_position_y_vec, rhs_position_y_vec);
				if constexpr (dims > 2)
					position_difference_z = hn::Sub(lhs_position_z_vec, rhs_position_z_vec);

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
					const simd_t repulsive_distance = hn::Add(lhs_radius_vec, rhs_radius_vec);

					if (!hn::AllTrue(tag_t(), hn::Gt(distance, repulsive_distance)))
					{
						repulsion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, repulsive_distance));
						repulsion = hn::Max(repulsion, hn::Zero(tag_t()));
						repulsion = hn::Mul(repulsion, repulsion);
						repulsion =
							hn::Mul(repulsion, hn::Sqrt(hn::Mul(lhs_repulsion_coeff_vec, rhs_repulsion_coeff_vec)));
					}
				}
				else
				{
					const simd_t repulsive_distance = hn::Add(lhs_radius_vec, rhs_radius_vec);

					repulsion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, repulsive_distance));
					repulsion = hn::Max(repulsion, hn::Zero(tag_t()));
					repulsion = hn::Mul(repulsion, repulsion);
					repulsion = hn::Mul(repulsion, hn::Sqrt(hn::Mul(lhs_repulsion_coeff_vec, rhs_repulsion_coeff_vec)));
				}

				// compute adhesion
				simd_t adhesion;
				if (try_skip_adhesion)
				{
					adhesion = hn::Zero(tag_t());
					simd_t tmp = hn::Mul(lhs_relative_maximum_adhesion_distance_vec, lhs_radius_vec);
					const simd_t adhesion_distance =
						hn::MulAdd(rhs_relative_maximum_adhesion_distance_vec, rhs_radius_vec, tmp);

					if (!hn::AllTrue(tag_t(), hn::Gt(distance, adhesion_distance)))
					{
						adhesion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, adhesion_distance));
						adhesion = hn::Max(adhesion, hn::Zero(tag_t()));
						adhesion = hn::Mul(adhesion, adhesion);

						const index_simd_t lhs_indices = hn::Iota(index_tag_t(), lhs);
						const index_simd_t rhs_indices = hn::Iota(index_tag_t(), rhs);
						const index_simd_t types_count = hn::Set(index_tag_t(), agent_types_count);

						index_simd_t lhs_index = hn::MulAdd(lhs_indices, types_count, rhs_agent_type_vec);
						index_simd_t rhs_index = hn::MulAdd(rhs_indices, types_count, lhs_agent_type_vec);

						simd_t lhs_adhesion_affinity_vec = hn::GatherIndex(tag_t(), lhs_adhesion_affinity, lhs_index);
						simd_t rhs_adhesion_affinity_vec = hn::GatherIndex(tag_t(), rhs_adhesion_affinity, rhs_index);

						adhesion = hn::Mul(
							adhesion, hn::Sqrt(hn::Mul(hn::Mul(lhs_adhesion_coeff_vec, rhs_adhesion_coeff_vec),
													   hn::Mul(lhs_adhesion_affinity_vec, rhs_adhesion_affinity_vec))));
					}
				}
				else
				{
					simd_t tmp = hn::Mul(lhs_relative_maximum_adhesion_distance_vec, lhs_radius_vec);
					const simd_t adhesion_distance =
						hn::MulAdd(rhs_relative_maximum_adhesion_distance_vec, rhs_radius_vec, tmp);

					adhesion = hn::Sub(hn::Set(tag_t(), 1), hn::Div(distance, adhesion_distance));
					adhesion = hn::Max(adhesion, hn::Zero(tag_t()));
					adhesion = hn::Mul(adhesion, adhesion);

					const index_simd_t lhs_indices = hn::Iota(index_tag_t(), lhs);
					const index_simd_t rhs_indices = hn::Iota(index_tag_t(), rhs);
					const index_simd_t types_count = hn::Set(index_tag_t(), agent_types_count);

					index_simd_t lhs_index = hn::MulAdd(lhs_indices, types_count, rhs_agent_type_vec);
					index_simd_t rhs_index = hn::MulAdd(rhs_indices, types_count, lhs_agent_type_vec);

					simd_t lhs_adhesion_affinity_vec = hn::GatherIndex(tag_t(), lhs_adhesion_affinity, lhs_index);
					simd_t rhs_adhesion_affinity_vec = hn::GatherIndex(tag_t(), rhs_adhesion_affinity, rhs_index);

					adhesion = hn::Mul(
						adhesion, hn::Sqrt(hn::Mul(hn::Mul(lhs_adhesion_coeff_vec, rhs_adhesion_coeff_vec),
												   hn::Mul(lhs_adhesion_affinity_vec, rhs_adhesion_affinity_vec))));
				}

				simd_t force = hn::Div(hn::Sub(repulsion, adhesion), distance);

				lhs_velocity_x_vec = hn::MulAdd(force, position_difference_x, lhs_velocity_x_vec);
				if constexpr (dims > 1)
					lhs_velocity_y_vec = hn::MulAdd(force, position_difference_y, lhs_velocity_y_vec);
				if constexpr (dims > 2)
					lhs_velocity_z_vec = hn::MulAdd(force, position_difference_z, lhs_velocity_z_vec);
			}
		}

		hn::StoreU(lhs_velocity_x_vec, tag_t(), lhs_velocity_x + lhs);
		if constexpr (dims > 1)
			hn::StoreU(lhs_velocity_y_vec, tag_t(), lhs_velocity_y + lhs);
		if constexpr (dims > 2)
			hn::StoreU(lhs_velocity_z_vec, tag_t(), lhs_velocity_z + lhs);
	}
}

template <std::size_t dims, typename real_t, typename index_t>
void reorder_agents(std::unique_ptr<voxel_data<real_t, index_t>[]>& grid,
					std::unique_ptr<agent_grid_idx<index_t>[]>& agent_indices, std::unique_ptr<std::mutex[]>& mutexes_,
					std::array<index_t, 3> grid_dims_, std::array<real_t, 3> voxel_size_, index_t agent_types_count_)
{
	// Reorder agents into voxels based on their grid indices
	// #pragma omp parallel for schedule(static) collapse(3)
	for (index_t x = 0; x < grid_dims_[0]; x++)
	{
		for (index_t y = 0; y < grid_dims_[1]; y++)
		{
			for (index_t z = 0; z < grid_dims_[2]; z++)
			{
				index_t voxel_idx = x + y * grid_dims_[0] + z * grid_dims_[0] * grid_dims_[1];

				voxel_data<real_t, index_t>& voxel = grid[voxel_idx];

				for (index_t local_idx = 0; local_idx < (index_t)voxel.agent_indices.size(); local_idx++)
				{
					// Calculate voxel index
					index_t voxel_x =
						static_cast<index_t>(std::max(voxel.positionsx[local_idx], (real_t)0) / voxel_size_[0]);
					index_t voxel_y =
						(dims > 1)
							? static_cast<index_t>(std::max(voxel.positionsy[local_idx], (real_t)0) / voxel_size_[1])
							: 0;
					index_t voxel_z =
						(dims > 2)
							? static_cast<index_t>(std::max(voxel.positionsz[local_idx], (real_t)0) / voxel_size_[2])
							: 0;

					// Clamp to grid bounds
					voxel_x = std::min(voxel_x, grid_dims_[0] - 1);
					voxel_y = std::min(voxel_y, grid_dims_[1] - 1);
					voxel_z = std::min(voxel_z, grid_dims_[2] - 1);

					// Convert to linear index
					index_t new_voxel_idx = voxel_x + voxel_y * grid_dims_[0] + voxel_z * grid_dims_[0] * grid_dims_[1];

					if (new_voxel_idx == voxel_idx)
						continue;

					std::scoped_lock lock(mutexes_[new_voxel_idx], mutexes_[voxel_idx]);

					voxel_data<real_t, index_t>& new_voxel = grid[new_voxel_idx];
					index_t new_local_idx = new_voxel.agent_indices.size();

					// Move agent data to new voxel
					new_voxel.agent_indices.push_back(voxel.agent_indices[local_idx]);

					agent_indices[voxel.agent_indices[local_idx]].voxel_idx = new_voxel_idx;
					agent_indices[voxel.agent_indices[local_idx]].local_idx = new_local_idx;

					new_voxel.positionsx.push_back(voxel.positionsx[local_idx]);
					if constexpr (dims > 1)
						new_voxel.positionsy.push_back(voxel.positionsy[local_idx]);
					if constexpr (dims > 2)
						new_voxel.positionsz.push_back(voxel.positionsz[local_idx]);
					new_voxel.velocitiesx.push_back(voxel.velocitiesx[local_idx]);
					if constexpr (dims > 1)
						new_voxel.velocitiesy.push_back(voxel.velocitiesy[local_idx]);
					if constexpr (dims > 2)
						new_voxel.velocitiesz.push_back(voxel.velocitiesz[local_idx]);
					new_voxel.radius.push_back(voxel.radius[local_idx]);
					new_voxel.repulsion_coeff.push_back(voxel.repulsion_coeff[local_idx]);
					new_voxel.adhesion_coeff.push_back(voxel.adhesion_coeff[local_idx]);
					new_voxel.max_adhesion_distance.push_back(voxel.max_adhesion_distance[local_idx]);
					for (index_t type_idx = 0; type_idx < agent_types_count_; ++type_idx)
					{
						new_voxel.adhesion_affinity.push_back(
							voxel.adhesion_affinity[local_idx * agent_types_count_ + type_idx]);
					}
					new_voxel.agent_types.push_back(voxel.agent_types[local_idx]);

					// Move last agent in current voxel to fill the gap
					index_t last_idx = voxel.agent_indices.size() - 1;
					if (local_idx != last_idx)
					{
						voxel.agent_indices[local_idx] = voxel.agent_indices[last_idx];

						agent_indices[voxel.agent_indices[local_idx]].voxel_idx = voxel_idx;
						agent_indices[voxel.agent_indices[local_idx]].local_idx = local_idx;

						voxel.positionsx[local_idx] = voxel.positionsx[last_idx];
						if constexpr (dims > 1)
							voxel.positionsy[local_idx] = voxel.positionsy[last_idx];
						if constexpr (dims > 2)
							voxel.positionsz[local_idx] = voxel.positionsz[last_idx];
						voxel.velocitiesx[local_idx] = voxel.velocitiesx[last_idx];
						if constexpr (dims > 1)
							voxel.velocitiesy[local_idx] = voxel.velocitiesy[last_idx];
						if constexpr (dims > 2)
							voxel.velocitiesz[local_idx] = voxel.velocitiesz[last_idx];
						voxel.radius[local_idx] = voxel.radius[last_idx];
						voxel.repulsion_coeff[local_idx] = voxel.repulsion_coeff[last_idx];
						voxel.adhesion_coeff[local_idx] = voxel.adhesion_coeff[last_idx];
						voxel.max_adhesion_distance[local_idx] = voxel.max_adhesion_distance[last_idx];
						for (index_t type_idx = 0; type_idx < agent_types_count_; ++type_idx)
						{
							voxel.adhesion_affinity[local_idx * agent_types_count_ + type_idx] =
								voxel.adhesion_affinity[last_idx * agent_types_count_ + type_idx];
						}
						voxel.agent_types[local_idx] = voxel.agent_types[last_idx];

						local_idx--; // Stay at the same index for next iteration
					}

					// Remove the last element (which was either moved or is the one we're relocating)
					voxel.agent_indices.pop_back();
					voxel.positionsx.pop_back();
					if constexpr (dims > 1)
						voxel.positionsy.pop_back();
					if constexpr (dims > 2)
						voxel.positionsz.pop_back();
					voxel.velocitiesx.pop_back();
					if constexpr (dims > 1)
						voxel.velocitiesy.pop_back();
					if constexpr (dims > 2)
						voxel.velocitiesz.pop_back();
					voxel.radius.pop_back();
					voxel.repulsion_coeff.pop_back();
					voxel.adhesion_coeff.pop_back();
					voxel.max_adhesion_distance.pop_back();
					for (index_t type_idx = 0; type_idx < agent_types_count_; ++type_idx)
					{
						voxel.adhesion_affinity.pop_back();
					}
					voxel.agent_types.pop_back();
				}
			}
		}
	}
}

template <std::size_t dims, typename real_t, typename index_t>
void solve_voxel_pair(voxel_data<real_t, index_t>& lhs, voxel_data<real_t, index_t>& rhs, index_t lhs_count,
					  index_t rhs_count, bool try_skip_repulsion, bool try_skip_adhesion, index_t agent_types_count)
{
	solve_pair<dims>(try_skip_repulsion, try_skip_adhesion, lhs_count, rhs_count, agent_types_count,
					 lhs.velocitiesx.data(), lhs.velocitiesy.data(), lhs.velocitiesz.data(), lhs.positionsx.data(),
					 lhs.positionsy.data(), lhs.positionsz.data(), lhs.radius.data(), lhs.repulsion_coeff.data(),
					 lhs.adhesion_coeff.data(), lhs.max_adhesion_distance.data(), lhs.adhesion_affinity.data(),
					 lhs.agent_types.data(), rhs.positionsx.data(), rhs.positionsy.data(), rhs.positionsz.data(),
					 rhs.radius.data(), rhs.repulsion_coeff.data(), rhs.adhesion_coeff.data(),
					 rhs.max_adhesion_distance.data(), rhs.adhesion_affinity.data(), rhs.agent_types.data());
}

template <typename real_t>
void transposed_grid_solver<real_t>::solve()
{
	for (index_t iter = 0; iter < iterations_; iter++)
	{
// Process voxel pairs for interactions
#pragma omp parallel for schedule(static) collapse(3)
		for (index_t x = 0; x < grid_dims_[0]; x++)
		{
			for (index_t y = 0; y < grid_dims_[1]; y++)
			{
				for (index_t z = 0; z < grid_dims_[2]; z++)
				{
					index_t voxel_idx = x + y * grid_dims_[0] + z * grid_dims_[0] * grid_dims_[1];
					voxel_data<real_t, index_t>& current_voxel = grid_[voxel_idx];
					index_t current_voxel_agent_count = current_voxel.agent_indices.size();

					// Iterate over neighboring voxels
					for (index_t dx = -1; dx <= 1; dx++)
					{
						index_t nx = x + dx;
						if (nx < 0 || nx >= grid_dims_[0])
							continue;

						for (index_t dy = -1; dy <= 1; dy++)
						{
							index_t ny = y + dy;
							if (ny < 0 || ny >= grid_dims_[1])
								continue;

							for (index_t dz = -1; dz <= 1; dz++)
							{
								index_t nz = z + dz;
								if (nz < 0 || nz >= grid_dims_[2])
									continue;

								index_t neighbor_idx = nx + ny * grid_dims_[0] + nz * grid_dims_[0] * grid_dims_[1];
								voxel_data<real_t, index_t>& neighbor_voxel = grid_[neighbor_idx];
								index_t neighbor_voxel_agent_count = neighbor_voxel.agent_indices.size();

								if (current_voxel_agent_count == 0 || neighbor_voxel_agent_count == 0)
									continue;

								// std::cout << "Processing voxel pair: (" << voxel_idx << ", " << neighbor_idx << ")"
								// 		  << " with agent counts: (" << current_voxel_agent_count << ", "
								// 		  << neighbor_voxel_agent_count << ")" << std::endl;

								// Process interaction between current and neighbor voxel
								if (dims_ == 1)
									solve_voxel_pair<1>(current_voxel, neighbor_voxel, current_voxel_agent_count,
														neighbor_voxel_agent_count, try_skip_repulsion_,
														try_skip_adhesion_, agent_types_count_);
								else if (dims_ == 2)
									solve_voxel_pair<2>(current_voxel, neighbor_voxel, current_voxel_agent_count,
														neighbor_voxel_agent_count, try_skip_repulsion_,
														try_skip_adhesion_, agent_types_count_);
								else if (dims_ == 3)
									solve_voxel_pair<3>(current_voxel, neighbor_voxel, current_voxel_agent_count,
														neighbor_voxel_agent_count, try_skip_repulsion_,
														try_skip_adhesion_, agent_types_count_);
							}
						}
					}
				}
			}
		}

		// Update positions based on velocities and copy data back
#pragma omp parallel for schedule(static)
		for (index_t voxel_idx = 0; voxel_idx < grid_dims_[0] * grid_dims_[1] * grid_dims_[2]; voxel_idx++)
		{
			voxel_data<real_t, index_t>& voxel = grid_[voxel_idx];
			index_t agent_count = voxel.agent_indices.size();

			for (index_t local_idx = 0; local_idx < agent_count; local_idx++)
			{
				voxel.positionsx[local_idx] += voxel.velocitiesx[local_idx] * timestep_;
				voxel.velocitiesx[local_idx] = 0;

				if (dims_ > 1)
				{
					voxel.positionsy[local_idx] += voxel.velocitiesy[local_idx] * timestep_;
					voxel.velocitiesy[local_idx] = 0;
				}

				if (dims_ > 2)
				{
					voxel.positionsz[local_idx] += voxel.velocitiesz[local_idx] * timestep_;
					voxel.velocitiesz[local_idx] = 0;
				}
			}
		}

		// Reorder agents into voxels based on updated positions
		if (dims_ == 1)
			reorder_agents<1, real_t, index_t>(grid_, agent_indices_, mutexes_, grid_dims_, voxel_size_,
											   agent_types_count_);
		else if (dims_ == 2)
			reorder_agents<2, real_t, index_t>(grid_, agent_indices_, mutexes_, grid_dims_, voxel_size_,
											   agent_types_count_);
		else if (dims_ == 3)
			reorder_agents<3, real_t, index_t>(grid_, agent_indices_, mutexes_, grid_dims_, voxel_size_,
											   agent_types_count_);
	}
}

template <typename real_t>
void transposed_grid_solver<real_t>::initialize(const nlohmann::json& params, const problem_t& problem)
{
	try_skip_repulsion_ = params.value("try_skip_repulsion", false);
	try_skip_adhesion_ = params.value("try_skip_adhesion", false);
	voxel_size_ = params.value("voxel_size", std::array<real_t, 3> { (real_t)30, (real_t)30, (real_t)30 });

	dims_ = static_cast<index_t>(problem.dims);
	timestep_ = static_cast<real_t>(problem.timestep);
	iterations_ = static_cast<index_t>(problem.iterations);
	agents_count_ = static_cast<index_t>(problem.agents_count);
	agent_types_count_ = static_cast<index_t>(problem.agent_types_count);
	domain_size_ = { static_cast<real_t>(problem.domain_size[0]), static_cast<real_t>(problem.domain_size[1]),
					 static_cast<real_t>(problem.domain_size[2]) };

	// Initialize agent distributor
	agent_distributor<real_t> distributor(problem);

	// Calculate grid dimensions based on domain_size_ and voxel_size_
	for (index_t i = 0; i < 3; ++i)
	{
		grid_dims_[i] = static_cast<index_t>(std::llround(std::ceil(domain_size_[i] / voxel_size_[i])));
		if (i >= dims_)
			grid_dims_[i] = 1;
	}
	index_t total_voxels = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];

	grid_ = std::make_unique<voxel_data<real_t, index_t>[]>(total_voxels);
	mutexes_ = std::make_unique<std::mutex[]>(total_voxels);
	agent_indices_ = std::make_unique<agent_grid_idx<index_t>[]>(agents_count_);

	// Distribute agents into grid based on their positions
	for (index_t agent_idx = 0; agent_idx < agents_count_; ++agent_idx)
	{
		// Get agent position
		real_t* pos = &distributor.positions_[agent_idx * dims_];

		// Calculate voxel index
		index_t voxel_x = static_cast<index_t>(std::max(pos[0], (real_t)0) / voxel_size_[0]);
		index_t voxel_y = (dims_ > 1) ? static_cast<index_t>(std::max(pos[1], (real_t)0) / voxel_size_[1]) : 0;
		index_t voxel_z = (dims_ > 2) ? static_cast<index_t>(std::max(pos[2], (real_t)0) / voxel_size_[2]) : 0;

		// Clamp to grid bounds
		voxel_x = std::min(voxel_x, grid_dims_[0] - 1);
		voxel_y = std::min(voxel_y, grid_dims_[1] - 1);
		voxel_z = std::min(voxel_z, grid_dims_[2] - 1);

		// Convert to linear index
		index_t voxel_idx = voxel_x + voxel_y * grid_dims_[0] + voxel_z * grid_dims_[0] * grid_dims_[1];

		// Get reference to the voxel
		auto& voxel = grid_[voxel_idx];

		// Store local index within voxel for this agent
		index_t local_idx = voxel.agent_indices.size();

		// Add agent data to voxel
		voxel.agent_indices.push_back(agent_idx);

		// Copy position data
		if (dims_ >= 1)
			voxel.positionsx.push_back(distributor.positions_[agent_idx * dims_ + 0]);
		if (dims_ >= 2)
			voxel.positionsy.push_back(distributor.positions_[agent_idx * dims_ + 1]);
		if (dims_ >= 3)
			voxel.positionsz.push_back(distributor.positions_[agent_idx * dims_ + 2]);

		// Initialize velocities to zero
		voxel.velocitiesx.push_back(0);
		if (dims_ > 1)
			voxel.velocitiesy.push_back(0);
		if (dims_ > 2)
			voxel.velocitiesz.push_back(0);

		// Copy physical properties
		voxel.radius.push_back(distributor.radius_[agent_idx]);
		voxel.repulsion_coeff.push_back(distributor.repulsion_coeff_[agent_idx]);
		voxel.adhesion_coeff.push_back(distributor.adhesion_coeff_[agent_idx]);
		voxel.max_adhesion_distance.push_back(distributor.max_adhesion_distance_[agent_idx]);

		// Copy adhesion affinity (entire row for this agent)
		for (index_t type_idx = 0; type_idx < agent_types_count_; ++type_idx)
		{
			voxel.adhesion_affinity.push_back(
				distributor.adhesion_affinity_[agent_idx * agent_types_count_ + type_idx]);
		}

		voxel.agent_types.push_back(distributor.agent_types_[agent_idx]);
		agent_indices_[agent_idx] = agent_grid_idx<index_t> { voxel_idx, local_idx };
	}
}

template <typename real_t>
std::array<double, 3> transposed_grid_solver<real_t>::access_agent(std::size_t agent_id)
{
	// Access agent data
	std::array<double, 3> agent_data = { 0.0, 0.0, 0.0 };
	auto& agent_idx = agent_indices_[agent_id];
	auto& voxel = grid_[agent_idx.voxel_idx];

	agent_data[0] = static_cast<double>(voxel.positionsx[agent_idx.local_idx]);
	if (dims_ > 1)
		agent_data[1] = static_cast<double>(voxel.positionsy[agent_idx.local_idx]);
	if (dims_ > 2)
		agent_data[2] = static_cast<double>(voxel.positionsz[agent_idx.local_idx]);
	return agent_data;
}

template <typename real_t>
void transposed_grid_solver<real_t>::save(std::ostream& os) const
{
	// Save agent data to output stream
	for (std::size_t i = 0; i < static_cast<std::size_t>(agents_count_); i++)
	{
		auto& agent_idx = agent_indices_[i];
		auto& voxel = grid_[agent_idx.voxel_idx];
		os << "Agent " << i << ": ";
		os << voxel.positionsx[agent_idx.local_idx] << " ";
		os << voxel.positionsy[agent_idx.local_idx] << " ";
		os << voxel.positionsz[agent_idx.local_idx] << " ";
		os << std::endl;
	}
}

template class transposed_grid_solver<float>;
template class transposed_grid_solver<double>;
