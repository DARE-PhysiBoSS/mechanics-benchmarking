#pragma once

#include <type_traits>

#include "../mechanics_solver.h"

template <typename real_t>
class transposed_solver : public mechanics_solver
{
	using index_t = std::conditional_t<std::is_same_v<real_t, float>, int32_t, int64_t>;

	std::unique_ptr<real_t[]> positionsx_;
	std::unique_ptr<real_t[]> positionsy_;
	std::unique_ptr<real_t[]> positionsz_;

	std::unique_ptr<real_t[]> velocitiesx_;
	std::unique_ptr<real_t[]> velocitiesy_;
	std::unique_ptr<real_t[]> velocitiesz_;

	std::unique_ptr<real_t[]> radius_;
	std::unique_ptr<real_t[]> repulsion_coeff_;
	std::unique_ptr<real_t[]> adhesion_coeff_;
	std::unique_ptr<real_t[]> max_adhesion_distance_;
	std::unique_ptr<real_t[]> adhesion_affinity_;
	std::unique_ptr<index_t[]> agent_types_;

	index_t dims_;
	real_t timestep_;
	index_t agents_count_;
	index_t agent_types_count_;

public:
	void solve() override;
	void initialize(const nlohmann::json& params, const problem_t& problem) override;
	std::array<double, 3> access_agent(std::size_t agent_id) override;
	void save(std::ostream& os) const override;
};
