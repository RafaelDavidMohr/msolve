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
#include <flint/ulong_extras.h>
#include "./msolve-data.h"
#include "../neogb/data.h"

typedef ulong CF_p; // for flint compatibility

/* Evaluation oracle: writes the values at the parameter point z of the
 * functions to interpolate into coeffs. Returns 0 on success, nonzero if
 * z is a bad point. */
typedef int (*evaluation_oracle_fn)(CF_p *coeffs, const CF_p *z, void *data);

/* Univariate polynomial in t over Fp(z_1, ..., z_r): the coefficient of
 * t^i is coeffs[2*i] / coeffs[2*i+1]. The nmod_mpoly_ctx_t is not stored. */
typedef struct{
  deg_t length;         /* number of coefficients, -1 if unset */
  deg_t alloc;          /* number of coefficients allocated */
  nmod_mpoly_t *coeffs; /* 2 * alloc entries */
} nmod_ratfunc_upoly_struct;
typedef nmod_ratfunc_upoly_struct nmod_ratfunc_upoly_t[1];

/* Rational parametrization over Fp(z_1, ..., z_r), as mpz_param_t but with
 * nmod_ratfunc_upoly_t polynomials, all over one nmod_mpoly_ctx_t (not
 * stored). */
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

/* Rational function with numerator sum_i t^degs_num[i] num[i] and
 * denominator 1 + sum_i t^degs_denom[i] denom[i], t a new variable. */
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

/* Precomputed data for univariate interpolation on the arguments
 * t_0, ..., t_{len-1}. Read-only after init, so it can be shared.
 * Uses FLINT internals (_nmod_poly_tree_*, _nmod_poly_interpolation_weights,
 * _nmod_poly_interpolate_nmod_vec_fast_precomp; written for FLINT 3.6). */
typedef struct {
  slong len;          /* number of arguments */
  nn_ptr *tree;       /* subproduct tree */
  nn_ptr weights;     /* interpolation weights */
  nmod_poly_t master; /* prod (t - t_i) */
} univariate_interp_data_struct;
typedef univariate_interp_data_struct univariate_interp_data_t[1];

/* Multivariate interpolation on a tensor grid (recursive Newton).
 * In:  grid[i], the degrees[i] + 1 arguments of grid variable i, which is
 *      variable nvars(ctx) - nvars + i of ctx; vals, the values on
 *      grid[0] x ... x grid[nvars-1] in row-major order (first variable
 *      slowest).
 * Out: interpolant, the polynomial in the grid variables with
 *      deg_i <= degrees[i] taking the values vals.
 * Assumes: 1 <= nvars <= nvars(ctx); arguments pairwise distinct; all
 *      entries reduced mod p; interpolant initialized with ctx. */
void
compute_newton_interpolant(const CF_p *vals, CF_p *const *grid,
			   const deg_t *degrees, int32_t nvars,
			   nmod_mpoly_t interpolant,
			   const nmod_mpoly_ctx_t ctx);

void univariate_interp_data_init(univariate_interp_data_t data,
                                 const CF_p *ts, slong len, nmod_t mod);

void univariate_interp_data_clear(univariate_interp_data_t data);

/* Rational function reconstruction with a known degree bound.
 * In:  data on N = 2d + 1 arguments t_i; vals[i] = f(t_i).
 * Out: num / denom = f as reduced fraction. Returns 0 if denom(0) != 0, then
 *      denom(0) = 1; returns 1 if denom(0) = 0, then num and denom are
 *      only defined up to a common constant.
 * Assumes: N odd, t_i pairwise distinct; f = P / D with deg P, deg D <= d
 *      and D(t_i) != 0; num, denom distinct, initialized with modulus p.
 * Uses FLINT internals (through data). */
int interpolate_univariate_ratfunc_from_data(const univariate_interp_data_t data,
					     const CF_p *vals,
					     nmod_poly_t num, nmod_poly_t denom);

/* Number of new points that must agree with a candidate to stop the
 * search below (zeta). Overridable with -DN_STABLE_POINTS_TO_STOP=... */
#ifndef N_STABLE_POINTS_TO_STOP
#define N_STABLE_POINTS_TO_STOP 2
#endif

/* Search for a univariate rational function of unknown degree, run on the
 * random linear combination f = sum_j lambda[j] c_j of ncoeffs functions. */
typedef struct {
  len_t ncoeffs;        /* number of functions c_j */
  CF_p *lambda;         /* random nonzero coefficients, drawn by init */
  nmod_poly_t F;        /* interpolant of f at all points */
  nmod_poly_t master;   /* prod (t - t_i) over all points */
  nmod_poly_t num, den; /* current candidate */
  int have_cand;        /* 1 if there is a current candidate */
  int cand_pole_at_0;   /* 1 if den(0) = 0, otherwise den(0) = 1 */
  slong npts;           /* number of points added */
  slong ncand;          /* number of points the candidate was computed from;
                         * the npts - ncand later points agreed with it */
} ratfunc_search_struct;
typedef ratfunc_search_struct ratfunc_search_t[1];

void ratfunc_search_init(ratfunc_search_t S, len_t ncoeffs, nmod_t mod,
                         flint_rand_t state);

void ratfunc_search_clear(ratfunc_search_t S);

/* In:  t; vals[j] = c_j(t) for j < ncoeffs.
 * Out: returns 1 if N_STABLE_POINTS_TO_STOP points (this one included)
 *      have agreed with the current candidate, 0 if more points are
 *      needed, -1 if t was added before (S unchanged).
 * Assumes: t and vals reduced mod p; no denominator of a c_j vanishes
 *      at t. */
int ratfunc_search_add_point(ratfunc_search_t S, CF_p t, const CF_p *vals);

/* Out: num / den, the current candidate. Returns 0 if den(0) = 1, 1 if
 *      den(0) = 0 (num and den only defined up to a common constant), -1
 *      if there is no candidate (num and den unchanged).
 * Assumes: num, den distinct, initialized with modulus p. */
int ratfunc_search_get(nmod_poly_t num, nmod_poly_t den,
                       const ratfunc_search_t S);


#endif
