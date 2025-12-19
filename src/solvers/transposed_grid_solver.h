#pragma once

#include <mutex>
#include <type_traits>

#include "../mechanics_solver.h"

template <typename real_t, typename index_t>
struct voxel_data
{
	std::vector<index_t> agent_indices;

	std::vector<real_t> positionsx;
	std::vector<real_t> positionsy;
	std::vector<real_t> positionsz;

	std::vector<real_t> velocitiesx;
	std::vector<real_t> velocitiesy;
	std::vector<real_t> velocitiesz;

	std::vector<real_t> radius;
	std::vector<real_t> repulsion_coeff;
	std::vector<real_t> adhesion_coeff;
	std::vector<real_t> max_adhesion_distance;
	std::vector<real_t> adhesion_affinity;
	std::vector<index_t> agent_types;
};

template <typename index_t>
struct agent_grid_idx
{
	index_t voxel_idx;
	index_t local_idx;
};

template <typename real_t>
class transposed_grid_solver : public mechanics_solver
{
	using index_t = std::conditional_t<std::is_same_v<real_t, float>, int32_t, int64_t>;

	std::unique_ptr<agent_grid_idx<index_t>[]> agent_indices_;
	std::unique_ptr<voxel_data<real_t, index_t>[]> grid_;
	std::unique_ptr<std::mutex[]> mutexes_;

	index_t dims_;
	real_t timestep_;
	index_t iterations_;
	index_t agents_count_;
	index_t agent_types_count_;
	std::array<real_t, 3> domain_size_;

	std::array<index_t, 3> grid_dims_;

	bool try_skip_repulsion_;
	bool try_skip_adhesion_;
	std::array<real_t, 3> voxel_size_;

public:
	void solve() override;
	void initialize(const nlohmann::json& params, const problem_t& problem) override;
	std::array<double, 3> access_agent(std::size_t agent_id) override;
	void save(std::ostream& os) const override;
};
