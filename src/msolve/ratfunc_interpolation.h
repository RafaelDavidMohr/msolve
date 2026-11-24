/* This file is part of msolve.
 *
 * msolve is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * msolve is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with msolve.  If not, see <https://www.gnu.org/licenses/>
 *
 * Authors:
 * Jérémy Berthomieu
 * Christian Eder
 * Rafael Mohr
 * Mohab Safey El Din */

/* This implementation is based on the rational function interpolation
 * implementation by Weijia Wang in AlgebraicSolving.jl (src/interp/), see
 * <https://github.com/algebraic-solving/AlgebraicSolving.jl>. */

#ifndef MSOLVE_RATFUNC_INTERPOLATION_H
#define MSOLVE_RATFUNC_INTERPOLATION_H

#include <flint/nmod.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_mpoly.h>
#include "./msolve-data.h"
#include "../neogb/data.h"

typedef ulong CF_p; // for flint compatibility

/* Multivariate interpolation on a tensor-product grid, by recursive Newton
 * interpolation (one variable at a time, the first grid variable outermost).
 *
 * Let p = ctx->mod.n and n_i = degrees[i] + 1. Computes the unique
 * polynomial f over Z/pZ such that
 *   - f only involves the last nvars variables of ctx: the i-th grid
 *     variable (0 <= i < nvars) is variable
 *     nmod_mpoly_ctx_nvars(ctx) - nvars + i of ctx,
 *   - deg_{x_i} f <= degrees[i] for every i,
 *   - f takes the values vals on grid[0] x grid[1] x ... x grid[nvars-1],
 * and stores it in interpolant.
 *
 * vals         the n_0 * n_1 * ... * n_{nvars-1} values, reduced mod p, in
 *              row-major order (first variable varying slowest): the value
 *              at (grid[0][e_0], ..., grid[nvars-1][e_{nvars-1}]) sits where
 *              it would in a C array vals[n_0][n_1]...[n_{nvars-1}].
 * grid         grid[i] holds the n_i nodes of the i-th grid variable,
 *              reduced mod p and pairwise distinct (so n_i <= p).
 * degrees      degree bounds, degrees[i] >= 0.
 * nvars        number of grid variables, 1 <= nvars <= nmod_mpoly_ctx_nvars(ctx).
 * interpolant  output; must be initialized with ctx. Its previous content is
 *              overwritten.
 * ctx          context of interpolant; p must be prime.
 *
 * vals, grid and degrees are not modified. The preconditions are not
 * checked: repeated nodes make FLINT abort ("Impossible inverse"), and
 * nodes or values that are not reduced mod p can give wrong results. */
void compute_newton_interpolant(const CF_p *vals, CF_p *const *grid, const deg_t *degrees,
				int32_t nvars,
				nmod_mpoly_t interpolant, const nmod_mpoly_ctx_t ctx);

#endif
