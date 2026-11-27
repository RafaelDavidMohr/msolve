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

/* black box for rational function reconstruction
   Writes values of length of coeffs rational functions
   evaluated at z into coeffs. Return 0 on success, nonzero
   if z is a bad point */
typedef int (*evaluation_oracle_fn)(CF_p *coeffs, const CF_p *z, void *data);

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
 * and len >= 1; for interpolate_univariate_ratfunc_from_data, len must be
 * odd. The preconditions are not checked. ts is not modified and is no
 * longer needed after the call. Free data with univariate_interp_data_clear. */
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
int interpolate_univariate_ratfunc_from_data(const univariate_interp_data_t data,
					     const CF_p *vals,
					     nmod_poly_t num, nmod_poly_t denom);

/* Number of consecutive new points that must agree with a candidate before
 * the search below stops (zeta in the notes). Each agreement of a wrong
 * candidate has probability about deg / p, so this should be at least 2,
 * and possibly more for small primes. Can be overridden at compile time,
 * e.g. with -DN_STABLE_POINTS_TO_STOP=3. */
#ifndef N_STABLE_POINTS_TO_STOP
#define N_STABLE_POINTS_TO_STOP 2
#endif

/* State of the search for a univariate rational function of unknown
 * degree (step 1 of the parametric algorithm).
 *
 * Points arrive one at a time. A point is an argument t_i together with
 * the values c_0(t_i), ..., c_{C-1}(t_i) of C rational functions
 * (C = ncoeffs). The search only looks at the random linear combination
 *   f = lambda[0] c_0 + ... + lambda[C-1] c_{C-1}
 * and stops once a candidate for f has agreed with
 * N_STABLE_POINTS_TO_STOP new points.
 *
 * How candidates are computed: when there is no candidate and the number
 * of points is odd, N = 2d + 1, hgcd on (master, F) gives a candidate
 * num / den with deg num <= d and deg den <= d. It is kept as long as it
 * agrees with every new point, and dropped at the first point where it
 * does not; the next candidate is then computed at the next odd number of
 * points. If max(deg P, deg D) = d for f = P / D in lowest terms, the
 * candidate computed at 2d + 1 points is f, so the search stops after
 * 2d + 1 + N_STABLE_POINTS_TO_STOP points, unless a wrong candidate
 * agreed with a point by chance (probability about d / p per point).
 *
 * The values of the c_j themselves are not stored: the caller keeps them
 * for the per-coefficient reconstruction. Once the search has stopped,
 * the first ncand points are the ones the candidate was computed from
 * (ncand is odd), and the last npts - ncand = N_STABLE_POINTS_TO_STOP
 * points are the ones that confirmed it. */
typedef struct {
  len_t ncoeffs;        /* C, number of values per point */
  CF_p *lambda;         /* the C coefficients of the combination, nonzero */
  nmod_poly_t F;        /* polynomial interpolating f at all points so far,
                         * deg F < npts; also holds the modulus */
  nmod_poly_t master;   /* prod (t - t_i) over all points so far */
  nmod_poly_t num, den; /* current candidate num / den, if have_cand */
  int have_cand;        /* 1 if there is a current candidate, 0 if not */
  int cand_pole_at_0;   /* 1 if den(0) = 0 (num, den not normalized), 0 if
                         * den(0) = 1; only meaningful if have_cand */
  slong npts;           /* number of points added so far */
  slong ncand;          /* number of points the current candidate was
                         * computed from; only meaningful if have_cand.
                         * Every point added after these agreed with the
                         * candidate, otherwise it would have been dropped,
                         * so npts - ncand points have confirmed it. */
} ratfunc_search_struct;
typedef ratfunc_search_struct ratfunc_search_t[1];

/* Initializes S for a search over Z/pZ, p = mod.n prime, with
 * ncoeffs >= 1 values per point. The coefficients lambda[0], ..., lambda[ncoeffs-1] of the
 * combination are drawn uniformly at random in [1, p-1] from state.
 * Free S with ratfunc_search_clear (also allowed directly after init). */
void ratfunc_search_init(ratfunc_search_t S, len_t ncoeffs, nmod_t mod,
                         flint_rand_t state);

/* Frees the memory used by S. */
void ratfunc_search_clear(ratfunc_search_t S);

/* Adds a point: the argument t and the ncoeffs values
 * vals[j] = c_j(t), all reduced mod p. The denominators of the c_j must
 * not vanish at t (a caller working with an oracle skips such points).
 * vals is not modified and can be reused after the call.
 *
 * Returns
 *    1  if the search has stopped: the current candidate was computed
 *       before this point, and at least N_STABLE_POINTS_TO_STOP points
 *       (this one included) have agreed with it since;
 *    0  if more points are needed;
 *   -1  if t was added before. Then the point is ignored and S is not
 *       modified.
 * Points can still be added after 1 has been returned; the return value
 * stays 1 as long as they agree with the candidate. */
int ratfunc_search_add_point(ratfunc_search_t S, CF_p t, const CF_p *vals);

/* Copies the current candidate for f into num / den. num and den must be
 * two different polynomials, initialized with modulus p; their previous
 * content is overwritten. Returns
 *    0  if den(0) != 0; then den(0) = 1 (normalized as in
 *       interpolate_univariate_ratfunc_from_data);
 *    1  if den(0) = 0; then num / den is the candidate, but num and den
 *       are only determined up to a common nonzero constant;
 *   -1  if there is no current candidate; then num and den are not
 *       modified.
 * This is meant to be called once add_point has returned 1, but it can be
 * called at any time. */
int ratfunc_search_get(nmod_poly_t num, nmod_poly_t den,
                       const ratfunc_search_t S);


#endif
