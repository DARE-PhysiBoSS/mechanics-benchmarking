// Example template: I will replace these with the real signatures from grid.h

#include "grid.h"

#include <cstdint>
#include <utility>



template <typename real_t>
std::size_t Grid<real_t>::get_grid_size() const
{
	return grid_size_x_ * grid_size_y_ * grid_size_z_;
}

template <typename real_t>
std::vector<std::size_t>& Grid<real_t>::get_agents_in_voxel(std::size_t voxel_index)
{
	return (grid_cells[voxel_index]);
}

template <typename real_t>
std::vector<std::size_t>& Grid<real_t>::get_moore_indices(std::size_t voxel_index)
{
	return (moore_neighbours[voxel_index]);
}

template <typename real_t>
void Grid<real_t>::create_moore_2d()
{
	moore_neighbours.resize(grid_cells.size());
	for (std::size_t i = 0; i < grid_cells.size(); ++i)
	{
		moore_neighbours[i].clear();
	}

	for (int x = 0; x < static_cast<int>(grid_size_x_); ++x)
	{
		for (int y = 0; y < static_cast<int>(grid_size_y_); ++y)
		{
			std::size_t center_index = y + x * grid_size_y_;

			for (int dx = -1; dx <= 1; ++dx)
			{
				for (int dy = -1; dy <= 1; ++dy)
				{
					if (dx == 0 && dy == 0)
						continue;

					int nx = x + dx;
					int ny = y + dy;

					if (nx >= 0 && nx < static_cast<int>(grid_size_x_) && ny >= 0
						&& ny < static_cast<int>(grid_size_y_))
					{
						std::size_t neighbour_index = ny + nx * grid_size_y_;

						if (neighbour_index < grid_cells.size())
						{
							moore_neighbours[center_index].push_back(neighbour_index);
						}
					}
				}
			}
		}
	}
}

template <typename real_t>
void Grid<real_t>::create_moore_3d()
{
	moore_neighbours.resize(grid_cells.size());
	for (std::size_t i = 0; i < grid_cells.size(); ++i)
	{
		moore_neighbours[i].clear();
	}

	for (int x = 0; x < static_cast<int>(grid_size_x_); ++x)
	{
		for (int y = 0; y < static_cast<int>(grid_size_y_); ++y)
		{
			for (int z = 0; z < static_cast<int>(grid_size_z_); ++z)
			{
				std::size_t center_index = z + y * grid_size_z_ + x * grid_size_z_ * grid_size_y_;

				for (int dx = -1; dx <= 1; ++dx)
				{
					for (int dy = -1; dy <= 1; ++dy)
					{
						for (int dz = -1; dz <= 1; ++dz)
						{
							if (dx == 0 && dy == 0 && dz == 0)
								continue;

							int nx = x + dx;
							int ny = y + dy;
							int nz = z + dz;

							if (nx >= 0 && nx < static_cast<int>(grid_size_x_) && ny >= 0
								&& ny < static_cast<int>(grid_size_y_) && nz >= 0
								&& nz < static_cast<int>(grid_size_z_))
							{
								std::size_t neighbour_index = nz + ny * grid_size_z_ + nx * grid_size_z_ * grid_size_y_;
								moore_neighbours[center_index].push_back(neighbour_index);
							}
						}
					}
				}
			}
		}
	}
}



template <typename real_t>
Grid<real_t>::Grid(std::vector<real_t> domain_size, std::vector<real_t> voxel_size)
{
	xside_ = domain_size[0];
	yside_ = domain_size[1];
	zside_ = domain_size.size() > 2 ? domain_size[2] : 1.0;

	is_2d_ = (domain_size.size() > 2) ? false : true;

	grid_size_x_ = static_cast<std::size_t>(std::ceil(xside_ / voxel_size[0]));
	grid_size_y_ = static_cast<std::size_t>(std::ceil(yside_ / voxel_size[1]));
	if (domain_size.size() > 2)
		grid_size_z_ = static_cast<std::size_t>(std::ceil(zside_ / voxel_size[2]));
	else
		grid_size_z_ = 1;

	dx_ = voxel_size[0];
	dy_ = voxel_size[1];
	dz_ = voxel_size.size() > 2 ? voxel_size[2] : 1.0;

	std::int64_t total_grid_size = static_cast<std::int64_t>(grid_size_x_) * static_cast<std::int64_t>(grid_size_y_)
								   * static_cast<std::int64_t>(grid_size_z_);
	grid_cells.resize(total_grid_size);
	for (auto& cell : grid_cells)
	{
		cell = std::vector<std::size_t>(0);
	}

	if (is_grid_2d())
	{
		create_moore_2d();
	}
	else
	{
		create_moore_3d();
	}
}

template <typename real_t>
Grid<real_t>::Grid()
{
	xside_ = 1.0;
	yside_ = 1.0;
	zside_ = 1.0;

	is_2d_ = true;

	grid_size_x_ = 1;
	grid_size_y_ = 1;
	grid_size_z_ = 1;

	dx_ = 1.0;
	dy_ = 1.0;
	dz_ = 1.0;

	grid_cells.resize(grid_size_x_ * grid_size_y_ * grid_size_z_);
	for (auto& cell : grid_cells)
	{
		cell = std::vector<std::size_t>(0);
	}

	if (is_grid_2d())
	{
		create_moore_2d();
	}
	else
	{
		create_moore_3d();
	}
}

template <typename real_t>
bool Grid<real_t>::is_grid_2d() const
{
	return is_2d_;
}

template <typename real_t>
std::size_t Grid<real_t>::voxel_index(std::vector<real_t> position)
{
	std::size_t index = 0;
	if (is_grid_2d())
	{
		std::size_t ix = static_cast<std::size_t>(position[0] / dx_);
		std::size_t iy = static_cast<std::size_t>(position[1] / dy_);
		index = iy + ix * grid_size_y_;
	}
	else
	{
		std::size_t ix = static_cast<std::size_t>(position[0] / dx_);
		std::size_t iy = static_cast<std::size_t>(position[1] / dy_);
		std::size_t iz = static_cast<std::size_t>(position[2] / dz_);
		index = iz + iy * grid_size_z_ + ix * grid_size_z_ * grid_size_y_;
	}
	if (index >= grid_cells.size())
		return grid_cells.size() - 1;
	return index;
}

template <typename real_t>
void Grid<real_t>::insert_agent(std::vector<real_t> position, std::size_t agent_id)
{
	std::size_t voxel_idx = voxel_index(position);
	grid_cells[voxel_idx].push_back(agent_id);
}

template <typename real_t>
std::vector<std::size_t> Grid<real_t>::get_grid_coordinates(std::vector<real_t> position)
{
	int dim = is_grid_2d() ? 2 : 3;
	std::vector<std::size_t> coords(dim, 0);
	coords[0] = static_cast<std::size_t>(position[0] / dx_);
	coords[1] = static_cast<std::size_t>(position[1] / dy_);
	if (!is_grid_2d())
	{
		coords[2] = static_cast<std::size_t>(position[2] / dz_);
	}
	return coords;
}

template <typename real_t>
std::vector<std::size_t> Grid<real_t>::get_grid_coordinates(std::size_t voxel_index)
{
	int dim = is_grid_2d() ? 2 : 3;
	std::vector<std::size_t> coords(dim, 0);

	if (is_grid_2d())
	{
		coords[1] = voxel_index % grid_size_y_;
		coords[0] = voxel_index / grid_size_y_;
	}
	else
	{
		coords[2] = voxel_index % grid_size_z_;
		std::size_t temp = voxel_index / grid_size_z_;
		coords[1] = temp % grid_size_y_;
		coords[0] = temp / (grid_size_y_);
	}

	return coords;
}

template <typename real_t>
Grid<real_t>::~Grid()
{
	/*
	for (auto cell : grid_cells)
	{
		delete cell;
	}
	for (auto neighbours : moore_neighbours)
	{
		delete neighbours;
	}*/
}



template class Grid<float>;
template class Grid<double>;
