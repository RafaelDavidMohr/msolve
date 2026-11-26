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

#ifndef MSOLVE_PARAMETRIC_H
#define MSOLVE_PARAMETRIC_H

#include <flint/nmod.h>
#include <flint/nmod_vec.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_mpoly.h>
#include "./msolve-data.h"
#include "../neogb/data.h"

typedef ulong CF_p; // for flint compatibility

/* univariate polynomial in t whose coefficients are rational functions
 * in Fp(z_1, ..., z_r), each with its own numerator and denominator (no
 * common denominator): the coefficient of t^i is
 * coeffs[2*i] / coeffs[2*i+1]. The numerators and denominators are
 * polynomials in z_1, ..., z_r over an nmod_mpoly_ctx_t that is not
 * stored here: the caller passes it to the functions that need it. */
typedef struct{
  deg_t length;         /* number of coefficients, -1 if unset */
  deg_t alloc;          /* number of coefficients allocated */
  nmod_mpoly_t *coeffs; /* 2 * alloc entries: the numerators at even
                         * indices, the denominators at odd indices */
} nmod_ratfunc_upoly_struct;
typedef nmod_ratfunc_upoly_struct nmod_ratfunc_upoly_t[1];

/* rational parametrization over Fp(z_1, ..., z_r), stored like
 * mpz_param_t except that its coefficients are rational functions in
 * z_1, ..., z_r, each with its own numerator and denominator (see
 * nmod_ratfunc_upoly_t): the coefficient of t^j in elim is
 * elim->coeffs[2*j] / elim->coeffs[2*j+1], and likewise for denom and
 * the coords[i]. All these polynomials use the same nmod_mpoly_ctx_t in
 * z_1, ..., z_r, which is not stored here. */
typedef struct{
  nvars_t nvars; /* number of variables */
  deg_t nsols;   /* degree of elim */
  len_t dquot;   /* dimension of the quotient ring */
  deg_t dim;     /* dimension of the ideal */
  nmod_ratfunc_upoly_t elim;
  nmod_ratfunc_upoly_t denom;
  nmod_ratfunc_upoly_t *coords; /* nvars - 1 coordinates */
} fp_param_with_parameters_struct;
typedef fp_param_with_parameters_struct fp_param_with_parameters_t[1];

// numerator: sum over t^i * num[i], denominator: 1 + sum over
// t^i*denom[i], t a new variable, 1 in the denominator not stored
typedef struct {

  int8_t degs_known;

  len_t num_num_terms;
  len_t den_num_terms;
  nmod_mpoly_t *num;
  nmod_mpoly_t *denom;
  deg_t *degs_num;
  deg_t *degs_denom;

  len_t num_num_vals;
  len_t den_num_vals;
  CF_p **num_vals;
  CF_p **denom_vals;
  
} rational_interpolant_struct;
typedef rational_interpolant_struct rational_interpolant_t[1];

/* precomputed data for univariate interpolation on fixed arguments
 * t_0, ..., t_{len-1}, shared by all reconstructions on these arguments (e.g.
 * all coefficients of a parametrization along one line). It is not
 * modified after univariate_interp_data_init, so it can also be shared
 * between threads.
 *
 * tree and weights are built with FLINT functions that are exported but
 * not part of FLINT's documented API (_nmod_poly_tree_*,
 * _nmod_poly_interpolation_weights,
 * _nmod_poly_interpolate_nmod_vec_fast_precomp), so they may change
 * between FLINT versions. Written against FLINT 3.6. */
typedef struct {
  slong len;          /* number of arguments */
  nn_ptr *tree;       /* subproduct tree of the arguments */
  nn_ptr weights;     /* interpolation weights, len entries */
  nmod_poly_t master; /* prod (t - t_i); also holds the modulus */
} univariate_interp_data_struct;
typedef univariate_interp_data_struct univariate_interp_data_t[1];

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
 *              where it would in a C array vals[n_0][n_1]...[n_{nvars-1}].
 * grid         grid[i] holds the n_i arguments of the i-th grid variable,
 *              reduced mod p and pairwise distinct (so n_i <= p).
 * degrees      degree bounds, degrees[i] >= 0.
 * nvars        number of grid variables,
 *              1 <= nvars <= nmod_mpoly_ctx_nvars(ctx).
 * interpolant  output; must be initialized with ctx. Its previous content
 *              is overwritten.
 * ctx          context of interpolant; p must be prime.
 *
 * vals, grid and degrees are not modified. The preconditions are not
 * checked: repeated arguments make FLINT abort ("Impossible inverse"), and
 * arguments or values that are not reduced mod p can give wrong results. */
void
compute_newton_interpolant(const CF_p *vals, CF_p *const *grid,
			   const deg_t *degrees, int32_t nvars,
			   nmod_mpoly_t interpolant,
			   const nmod_mpoly_ctx_t ctx);

/* Initializes data for the len arguments ts[0], ..., ts[len-1] modulo the
 * prime p = mod.n. The arguments must be reduced mod p and pairwise distinct,
 * and len >= 1; for interpolate_univariate_ratfunc, len must be odd. The
 * preconditions are not checked. ts is not modified and is no longer
 * needed after the call. Free data with univariate_interp_data_clear. */
void univariate_interp_data_init(univariate_interp_data_t data,
                                 const CF_p *ts, slong len, nmod_t mod);

/* Frees the memory used by data. */
void univariate_interp_data_clear(univariate_interp_data_t data);

/* Univariate rational function interpolation with a known degree bound,
 * on precomputed arguments: polynomial interpolation followed by FLINT's
 * half-gcd (Cauchy interpolation).
 *
 * Let p be the modulus of data, t_0, ..., t_{N-1} its arguments and
 * N = data->len = 2 * deg + 1 (N must be odd). Suppose vals[i] = f(t_i)
 * for a rational function f = P / D over Z/pZ with deg P <= deg and
 * deg D <= deg. Then this computes f = num / denom in lowest terms.
 *
 * Returns 0 if D(0) != 0. Then num / denom is normalized so that the
 * denominator has constant term 1: denom(0) = 1, i.e. num = p and
 * denom = 1 + q in the form p / (1 + q). If f = 0, the result is num = 0,
 * denom = 1.
 *
 * Returns 1 if D(0) = 0, so that f cannot be written as p / (1 + q). Then
 * num / denom = f still holds, but num and denom are not normalized: they
 * are only determined up to a common nonzero constant factor.
 *
 * data   precomputed data for the N arguments. It is not modified, so the
 *        same data can be used for many calls (and by several threads).
 * vals   the N values f(t_i), reduced mod p. D must not vanish at the
 *        arguments.
 * num    output, the numerator.
 * denom  output, the denominator; denom(0) = 1 if 0 is returned.
 *
 * Initialization: num and denom must be two different polynomials, already
 * initialized by the caller with modulus p. Their previous content is
 * overwritten, and the caller clears them afterwards. vals is not
 * modified.
 *
 * The preconditions are not checked. If the values do not come from such
 * an f, the result is unspecified, and FLINT may abort. */
int interpolate_univariate_ratfunc(const univariate_interp_data_t data,
				   const CF_p *vals,
				   nmod_poly_t num, nmod_poly_t denom);

/* Univariate rational function interpolation with a known degree bound,
 * without precomputation: builds the data for the arguments args, calls
 * interpolate_univariate_ratfunc and frees the data again. When several
 * reconstructions use the same arguments, call univariate_interp_data_init
 * once and interpolate_univariate_ratfunc for each of them instead.
 *
 * Let p = chr and N = 2 * deg + 1. Suppose vals[i] = f(args[i]) for a
 * rational function f = P / D over Z/pZ with deg P <= deg and
 * deg D <= deg. Then this computes f = num / denom in lowest terms. The
 * return value and the normalization are as for
 * interpolate_univariate_ratfunc:
 *   - 0 if D(0) != 0: denom(0) = 1, i.e. num = p and denom = 1 + q in the
 *     form p / (1 + q). If f = 0, the result is num = 0, denom = 1.
 *   - 1 if D(0) = 0: num / denom = f, but num and denom are not
 *     normalized (only determined up to a common nonzero constant).
 *
 * args   the N arguments, reduced mod p and pairwise distinct (so N <= p).
 * vals   the N values f(args[i]), reduced mod p. D must not vanish at the
 *        arguments.
 * deg    a bound on the degrees of both P and D, deg >= 0. It need not be
 *        tight.
 * num    output, the numerator.
 * denom  output, the denominator; denom(0) = 1 if 0 is returned.
 * chr    the prime p.
 *
 * Initialization: num and denom must be two different polynomials, already
 * initialized by the caller with modulus chr (e.g. nmod_poly_init(num,
 * chr)); FLINT uses the modulus stored in them. Their previous content is
 * overwritten, and the caller clears them afterwards. args and vals are
 * not modified.
 *
 * The preconditions are not checked. If the values do not come from such
 * an f, the result is unspecified, and FLINT may abort. */
int compute_univariate_ratfunc_known_deg(const CF_p *args, const CF_p *vals, const deg_t deg,
					 nmod_poly_t num, nmod_poly_t denom,
					 CF_p chr);

#endif
