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

#define RATFUNC_DEGREE_UNKNOWN -2

typedef ulong CF_p; // for flint compatibility

// numerator: sum over t^i * num[i], denominator: 1 + sum over t^i*denom[i], t a new variable
typedef struct {
  nmod_mpoly_t *num;
  nmod_mpoly_t *denom;
  deg_t deg_num;
  deg_t deg_denom;
} rational_interpolant_struct;
typedef rational_interpolant_struct rational_interpolant_t[1];

/* Multivariate interpolation on a tensor-product grid, by recursive Newton
 * interpolation (one variable at a time, the first grid variable
 * outermost).
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
 *              at (grid[0][e_0], ..., grid[nvars-1][e_{nvars-1}]) sits
 * where it would in a C array vals[n_0][n_1]...[n_{nvars-1}]. grid grid[i]
 * holds the n_i nodes of the i-th grid variable, reduced mod p and pairwise
 * distinct (so n_i <= p). degrees      degree bounds, degrees[i] >= 0.
 * nvars        number of grid variables, 1 <= nvars <=
 * nmod_mpoly_ctx_nvars(ctx). interpolant  output; must be initialized with
 * ctx. Its previous content is overwritten. ctx          context of
 * interpolant; p must be prime.
 *
 * vals, grid and degrees are not modified. The preconditions are not
 * checked: repeated nodes make FLINT abort ("Impossible inverse"), and
 * nodes or values that are not reduced mod p can give wrong results. */
void
compute_newton_interpolant(const CF_p *vals, CF_p *const *grid,
			   const deg_t *degrees, int32_t nvars,
			   nmod_mpoly_t interpolant,
			   const nmod_mpoly_ctx_t ctx);

/* Univariate rational function interpolation with a known degree bound:
 * polynomial interpolation followed by FLINT's half-gcd (Cauchy
 * interpolation).
 *
 * Let p = chr and N = 2 * deg + 1. Suppose vals[i] = f(args[i]) for a
 * rational function f = P / D over Z/pZ with deg P <= deg, deg D <= deg
 * and D(0) != 0. Then this computes f in lowest terms, normalized so that
 * the denominator has constant term 1: f = num / denom with denom(0) = 1,
 * i.e. num = p and denom = 1 + q in the form p / (1 + q). If f = 0, the
 * result is num = 0, denom = 1.
 *
 * args   the N nodes, reduced mod p and pairwise distinct (so N <= p).
 * vals   the N values f(args[i]), reduced mod p. D must not vanish at the
 *        nodes.
 * deg    a bound on the degrees of both P and D, deg >= 0. It need not be
 *        tight.
 * num    output, the numerator.
 * denom  output, the denominator, with denom(0) = 1.
 * chr    the prime p.
 *
 * Initialization: num and denom must be two different polynomials, already
 * initialized by the caller with modulus chr (e.g. nmod_poly_init(num,
 * chr)); FLINT uses the modulus stored in them. Their previous content is
 * overwritten, and the caller clears them afterwards. args and vals are
 * not modified.
 *
 * The preconditions are not checked. If D(0) = 0, FLINT aborts
 * ("Impossible inverse"). If the values do not come from such an f, the
 * result is unspecified, and FLINT may also abort. */
void compute_univariate_ratfunc_known_deg(const CF_p *args, const CF_p *vals, const deg_t deg,
					  nmod_poly_t num, nmod_poly_t denom,
					  CF_p chr);

#endif
