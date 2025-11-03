#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "mechanics_solver.h"
#include "problem.h"

template <typename real_t>
class Grid
{
private:
	std::vector<std::vector<std::size_t>> grid_cells;

	std::vector<std::vector<std::size_t>> moore_neighbours;

	real_t xside_, yside_, zside_;

	std::size_t grid_size_x_;
	std::size_t grid_size_y_;
	std::size_t grid_size_z_;

	real_t dx_, dy_, dz_;

	bool is_2d_;

public:
	Grid();
	Grid(std::vector<real_t> domain_size, std::vector<real_t> voxel_size);

	void insert_agent(std::vector<real_t> position, std::size_t agent_id);

	const std::vector<std::size_t>& get_agents_in_cell(std::size_t cell_x, std::size_t cell_y) const;

	std::size_t get_grid_size_x() const;

	std::size_t get_grid_size_y() const;

	std::size_t get_grid_size_z() const;

	std::size_t get_grid_size() const;

	std::vector<std::size_t>& get_agents_in_voxel(std::size_t voxel_index);

	std::vector<std::size_t>& get_moore_indices(std::size_t voxel_index);

	bool is_grid_2d() const;

	std::size_t voxel_index(std::vector<real_t> position);

	std::vector<std::size_t> get_grid_coordinates(std::vector<real_t> position);

	std::vector<std::size_t> get_grid_coordinates(std::size_t voxel_index);


	void create_moore_2d();
	void create_moore_3d();

	~Grid();
};
