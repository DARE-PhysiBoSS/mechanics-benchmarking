#pragma once

#include "../mechanics_solver.h"

template <typename real_t>
class base_solver : public mechanics_solver
{
	using index_t = int32_t;

	std::unique_ptr<real_t[]> positions_;
	std::unique_ptr<real_t[]> velocities_;
	std::unique_ptr<real_t[]> radius_;
	std::unique_ptr<real_t[]> repulsion_coeff_;
	std::unique_ptr<real_t[]> adhesion_coeff_;
	std::unique_ptr<real_t[]> max_adhesion_distance_;
	std::unique_ptr<real_t[]> adhesion_affinity_;
	std::unique_ptr<index_t[]> agent_types_;

	index_t dims_;
	index_t agents_count_;
	index_t agent_types_count_;

	bool use_symmetry_;
	bool try_skip_repulsion_;

public:
	void solve() override;
	void initialize(const nlohmann::json& params, const problem_t& problem) override;
	std::array<double, 3> access_agent(std::size_t agent_id) override;
	void save(std::ostream& os) const override;
};
