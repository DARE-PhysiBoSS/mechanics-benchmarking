#pragma once

#include <cmath>

template <std::size_t dims>
struct position_helper
{};

template <>
struct position_helper<1>
{
	template <typename real_t>
	static constexpr real_t distance(const real_t* __restrict__ lhs, const real_t* __restrict__ rhs)
	{
		return std::abs(lhs[0] - rhs[0]);
	}

	template <typename real_t>
	static constexpr real_t difference_and_distance(const real_t* __restrict__ lhs, const real_t* __restrict__ rhs,
													real_t* __restrict__ difference)
	{
		difference[0] = lhs[0] - rhs[0];

		return std::abs(difference[0]);
	}

	template <typename real_t>
	static constexpr void update_velocities(real_t* __restrict__ lhs, real_t* __restrict__ rhs,
											const real_t* __restrict__ difference, const real_t force)
	{
		lhs[0] += force * difference[0];
		rhs[0] -= force * difference[0];
	}

	template <typename real_t>
	static constexpr void update_velocity(real_t* __restrict__ velocity, const real_t* __restrict__ difference,
										  const real_t force)
	{
		velocity[0] += force * difference[0];
	}

	template <typename real_t>
	static constexpr void add(real_t* __restrict__ lhs, const real_t* __restrict__ rhs)
	{
		lhs[0] += rhs[0];
	}

	template <typename real_t>
	static constexpr void subtract(real_t* __restrict__ dst, const real_t* __restrict__ lhs,
								   const real_t* __restrict__ rhs)
	{
		dst[0] = lhs[0] - rhs[0];
	}
};

template <>
struct position_helper<2>
{
	template <typename real_t>
	static constexpr real_t distance(const real_t* __restrict__ lhs, const real_t* __restrict__ rhs)
	{
		return std::sqrt((lhs[0] - rhs[0]) * (lhs[0] - rhs[0]) + (lhs[1] - rhs[1]) * (lhs[1] - rhs[1]));
	}

	template <typename real_t>
	static constexpr real_t difference_and_distance(const real_t* __restrict__ lhs, const real_t* __restrict__ rhs,
													real_t* __restrict__ difference)
	{
		difference[0] = lhs[0] - rhs[0];
		difference[1] = lhs[1] - rhs[1];

		return std::sqrt(difference[0] * difference[0] + difference[1] * difference[1]);
	}

	template <typename real_t>
	static constexpr void update_velocities(real_t* __restrict__ lhs, real_t* __restrict__ rhs,
											const real_t* __restrict__ difference, const real_t force)
	{
		lhs[0] += force * difference[0];
		lhs[1] += force * difference[1];

		rhs[0] -= force * difference[0];
		rhs[1] -= force * difference[1];
	}

	template <typename real_t>
	static constexpr void update_velocity(real_t* __restrict__ velocity, const real_t* __restrict__ difference,
										  const real_t force)
	{
		velocity[0] += force * difference[0];
		velocity[1] += force * difference[1];
	}

	template <typename real_t>
	static constexpr void add(real_t* __restrict__ lhs, const real_t* __restrict__ rhs)
	{
		lhs[0] += rhs[0];
		lhs[1] += rhs[1];
	}

	template <typename real_t>
	static constexpr void subtract(real_t* __restrict__ dst, const real_t* __restrict__ lhs,
								   const real_t* __restrict__ rhs)
	{
		dst[0] = lhs[0] - rhs[0];
		dst[1] = lhs[1] - rhs[1];
	}
};

template <>
struct position_helper<3>
{
	template <typename real_t>
	static constexpr real_t distance(const real_t* __restrict__ lhs, const real_t* __restrict__ rhs)
	{
		return std::sqrt((lhs[0] - rhs[0]) * (lhs[0] - rhs[0]) + (lhs[1] - rhs[1]) * (lhs[1] - rhs[1])
						 + (lhs[2] - rhs[2]) * (lhs[2] - rhs[2]));
	}

	template <typename real_t>
	static constexpr real_t difference_and_distance(const real_t* __restrict__ lhs, const real_t* __restrict__ rhs,
													real_t* __restrict__ difference)
	{
		difference[0] = lhs[0] - rhs[0];
		difference[1] = lhs[1] - rhs[1];
		difference[2] = lhs[2] - rhs[2];

		return std::sqrt(difference[0] * difference[0] + difference[1] * difference[1] + difference[2] * difference[2]);
	}

	template <typename real_t>
	static constexpr void update_velocities(real_t* __restrict__ lhs, real_t* __restrict__ rhs,
											const real_t* __restrict__ difference, const real_t force)
	{
		lhs[0] += force * difference[0];
		lhs[1] += force * difference[1];
		lhs[2] += force * difference[2];

		rhs[0] -= force * difference[0];
		rhs[1] -= force * difference[1];
		rhs[2] -= force * difference[2];
	}

	template <typename real_t>
	static constexpr void update_velocity(real_t* __restrict__ velocity, const real_t* __restrict__ difference,
										  const real_t force)
	{
		velocity[0] += force * difference[0];
		velocity[1] += force * difference[1];
		velocity[2] += force * difference[2];
	}

	template <typename real_t>
	static constexpr void add(real_t* __restrict__ lhs, const real_t* __restrict__ rhs)
	{
		lhs[0] += rhs[0];
		lhs[1] += rhs[1];
		lhs[2] += rhs[2];
	}

	template <typename real_t>
	static constexpr void subtract(real_t* __restrict__ dst, const real_t* __restrict__ lhs,
								   const real_t* __restrict__ rhs)
	{
		dst[0] = lhs[0] - rhs[0];
		dst[1] = lhs[1] - rhs[1];
		dst[2] = lhs[2] - rhs[2];
	}
};
