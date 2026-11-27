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

#include "./msolve_parametric.h"

/* helpers for parametrizations with parameters */

/* initializes poly as an empty univariate polynomial with coefficients
 * in Fp(z_1, ..., z_r): no coefficients allocated and, as for
 * mpz_upoly_t, length -1 (unset) */
static inline void nmod_ratfunc_upoly_init(nmod_ratfunc_upoly_t poly) {
  poly->length = -1;
  poly->alloc = 0;
  poly->coeffs = NULL;
}

/* frees the memory used by poly, assuming that the 2 * alloc entries
 * coeffs[0], ..., coeffs[2*alloc-1] (the numerators and denominators of
 * the alloc coefficients) have been initialized with ctx */
static inline void nmod_ratfunc_upoly_clear(nmod_ratfunc_upoly_t poly,
                                            const nmod_mpoly_ctx_t ctx) {
  for (deg_t i = 0; i < 2 * poly->alloc; i++) {
    nmod_mpoly_clear(poly->coeffs[i], ctx);
  }
  free(poly->coeffs);
}

/* initializes param as an empty parametrization: no variables and no
 * coordinates */
static inline void fp_param_with_parameters_init(
    fp_param_with_parameters_t param) {
  param->nvars = 0;
  param->nsols = 0;
  param->dquot = 0;
  param->dim = 0;
  nmod_ratfunc_upoly_init(param->elim);
  nmod_ratfunc_upoly_init(param->denom);
  param->coords = NULL;
}

/* frees the memory used by param, whose polynomials have been
 * initialized with ctx. If coords is not NULL, its nvars - 1 coordinates
 * are assumed to be initialized. */
static inline void fp_param_with_parameters_clear(
    fp_param_with_parameters_t param, const nmod_mpoly_ctx_t ctx) {
  nmod_ratfunc_upoly_clear(param->elim, ctx);
  nmod_ratfunc_upoly_clear(param->denom, ctx);
  if (param->coords != NULL) {
    for (nvars_t i = 0; i < param->nvars - 1; i++) {
      nmod_ratfunc_upoly_clear(param->coords[i], ctx);
    }
  }
  free(param->coords);
}

/* helpers for rational interpolants */

static inline void rational_interpolant_init(rational_interpolant_t interp) {

  interp->degs_known = 0;

  interp->num_num_terms = 0;
  interp->den_num_terms = 0;
  interp->degs_denom = NULL;
  interp->degs_num = NULL;
  interp->denom = NULL;
  interp->num = NULL;

  interp->num_num_vals = 0;
  interp->den_num_vals = 0;
  interp->num_vals = NULL;
  interp->denom_vals = NULL;

  return;
}

static inline void rational_interpolant_clear(rational_interpolant_t interp,
					      const nmod_mpoly_ctx_t ctx) {

  for (len_t i = 0; i < interp->num_num_terms; i++) {
    nmod_mpoly_clear(interp->num[i], ctx);
  }
  for (len_t i = 0; i < interp->den_num_terms; i++) {
    nmod_mpoly_clear(interp->denom[i], ctx);
  }
  free(interp->num);
  free(interp->denom);
  free(interp->degs_num);
  free(interp->degs_denom);

  for (len_t i = 0; i < interp->num_num_vals; i++) {
    free(interp->num_vals[i]);
  }
  free(interp->num_vals);
  for (len_t i = 0; i < interp->den_num_vals; i++) {
    free(interp->denom_vals[i]);
  }
  free(interp->denom_vals);

  return;
}

static inline void
rational_interpolant_declare_deg(rational_interpolant_t interp,
				 len_t num_num_terms, deg_t *deg_num, 
				 len_t den_num_terms, deg_t *deg_denom,
				 const nmod_mpoly_ctx_t ctx) {

  interp->degs_known = 1;
  interp->num_num_terms = num_num_terms;
  interp->den_num_terms = den_num_terms;
  interp->degs_num = deg_num;
  interp->degs_denom = deg_denom;

  /* malloc(0) may return NULL, which is not an error */
  interp->num = malloc(num_num_terms * sizeof(nmod_mpoly_t));
  if (interp->num == NULL && num_num_terms > 0) {
    fprintf(stderr, "Error when allocating for rational function.\n");
    exit(1);
  }
  for (len_t i = 0; i < interp->num_num_terms; i++) {
    nmod_mpoly_init(interp->num[i], ctx);
  }

  interp->denom = malloc(den_num_terms * sizeof(nmod_mpoly_t));
  if (interp->denom == NULL && den_num_terms > 0) {
    fprintf(stderr, "Error when allocating for rational function.\n");
    exit(1);
  }
  for (len_t i = 0; i < interp->den_num_terms; i++) {
    nmod_mpoly_init(interp->denom[i], ctx);
  }
  return;
}

/* Sets t = 1 in interp: num = num[0] + ... + num[num_num_terms - 1] and
 * denom = 1 + denom[0] + ... + denom[den_num_terms - 1], so that
 * num / denom is the interpolated rational function in the variables of
 * ctx. Nothing is reduced or normalized beyond that. If no terms have
 * been declared, the result is 0 / 1.
 *
 * num and denom must be two different polynomials, not stored in interp
 * and already initialized with ctx. Their previous content is
 * overwritten. interp is not modified. */
static inline void
rational_interpolant_get_ratfunc(const rational_interpolant_t interp,
                                 nmod_mpoly_t num, nmod_mpoly_t denom,
                                 const nmod_mpoly_ctx_t ctx) {

  nmod_mpoly_zero(num, ctx);
  for (len_t i = 0; i < interp->num_num_terms; i++) {
    nmod_mpoly_add(num, num, interp->num[i], ctx);
  }

  nmod_mpoly_one(denom, ctx);
  for (len_t i = 0; i < interp->den_num_terms; i++) {
    nmod_mpoly_add(denom, denom, interp->denom[i], ctx);
  }
  return;
}

/* Multivariate interpolation */

void compute_newton_interpolant(const CF_p *vals, CF_p *const *grid, const deg_t *degrees,
				int32_t nvars,
                                nmod_mpoly_t interpolant, const nmod_mpoly_ctx_t ctx) {

  slong var = nmod_mpoly_ctx_nvars(ctx) - nvars;
  nmod_t mod = ctx->mod;
  
  if (nvars == 1) {

    nmod_poly_t rres;
    nmod_poly_init(rres, mod.n);
    nmod_poly_interpolate_nmod_vec(rres, grid[0], vals, degrees[0] + 1);
    nmod_mpoly_set_nmod_poly(interpolant, rres, var, ctx);
    nmod_poly_clear(rres);
    return;

  }

  nmod_mpoly_t master;
  nmod_mpoly_init(master, ctx);
  nmod_mpoly_set_ui(master, 1, ctx);

  slong point_count_next_var = 1;
  for (int i = 1; i < nvars; i++) {
    point_count_next_var *= (degrees[i] + 1);
  }

  nmod_mpoly_t correction, summand, lin_fac;
  nmod_mpoly_init(correction, ctx);
  nmod_mpoly_init(summand, ctx);
  nmod_mpoly_init(lin_fac, ctx);

  const CF_p *varvals = grid[0];
  slong val_ind = 0;
  nmod_mpoly_zero(interpolant, ctx); // this may need to go at some point if we want to keep updating
  for (int v = 0; v < degrees[0] + 1; v++) {
    CF_p varval = varvals[v];
    compute_newton_interpolant(&(vals[val_ind]), &(grid[1]),
			       &(degrees[1]), nvars-1,
			       correction, ctx);
    val_ind += point_count_next_var;

    nmod_mpoly_evaluate_one_ui(summand, interpolant, var, varval, ctx);
    nmod_mpoly_sub(summand, correction, summand, ctx);

    CF_p mval = 1;
    for (int j = 0; j < v; j++) {
      mval =
          nmod_mul(mval, nmod_sub(varval, varvals[j], mod), mod);
    }

    nmod_mpoly_scalar_mul_ui(summand, summand, nmod_inv(mval, mod), ctx);
    nmod_mpoly_mul(summand, summand, master, ctx);

    nmod_mpoly_add(interpolant, interpolant, summand, ctx);

    nmod_mpoly_gen(lin_fac, var, ctx);
    nmod_mpoly_sub_ui(lin_fac, lin_fac, varval, ctx);
    nmod_mpoly_mul(master, master, lin_fac, ctx);
  }

  nmod_mpoly_clear(correction, ctx);
  nmod_mpoly_clear(master, ctx);
  nmod_mpoly_clear(summand, ctx);
  nmod_mpoly_clear(lin_fac, ctx);

  return;
}

/* Univariate rational function interpolation */

void univariate_interp_data_init(univariate_interp_data_t data,
                                 const CF_p *ts, slong len, nmod_t mod) {

  data->len = len;

  data->tree = _nmod_poly_tree_alloc(len);
  _nmod_poly_tree_build(data->tree, ts, len, mod);

  /* functions starting with _ do not allocate their output */
  data->weights = _nmod_vec_init(len);
  _nmod_poly_interpolation_weights(data->weights, data->tree, len, mod);

  nmod_poly_init_mod(data->master, mod);
  nmod_poly_product_roots_nmod_vec(data->master, ts, len);

  return;
}

void univariate_interp_data_clear(univariate_interp_data_t data) {

  _nmod_poly_tree_free(data->tree, data->len);
  _nmod_vec_clear(data->weights);
  nmod_poly_clear(data->master);
  return;
}

/* Rational function reconstruction from an interpolating polynomial, by
 * hgcd on (master, F). Used by interpolate_univariate_ratfunc_from_data
 * and by the search (ratfunc_search_add_point).
 *
 * master = prod (t - t_i) over N = 2d + 1 distinct arguments (N odd), and
 * F is the polynomial of degree < N interpolating the values at the t_i;
 * F must be normalized (as all nmod_poly functions keep it). hgcd stops at
 * the first remainder of degree <= d, so the result num / denom always
 * has deg num <= d and deg denom <= d. If the values come from f = P / D
 * with deg P, deg D <= d and D(t_i) != 0, then num / denom = f in lowest
 * terms. Return value and normalization as for
 * interpolate_univariate_ratfunc_from_data (0: denom(0) = 1; 1: denom(0)
 * = 0, num and denom not normalized). num and denom must be two different
 * polynomials, different from F and master, initialized with the modulus
 * of F; their previous content is overwritten. */
static int interpolate_univariate_ratfunc(const nmod_poly_t F,
                                          const nmod_poly_t master,
                                          nmod_poly_t num, nmod_poly_t denom) {

  nmod_t mod = F->mod;
  /* outputs of hgcd that we use: denom is m11, num is B;
   * m12, m21, m22 and A are only workspace */
  nmod_poly_t m12, m21, m22, A;
  nmod_poly_init_mod(m12, mod);
  nmod_poly_init_mod(m21, mod);
  nmod_poly_init_mod(m22, mod);
  nmod_poly_init_mod(A, mod);

  slong sgn;
  sgn = nmod_poly_hgcd(denom, m12, m21, m22, A, num, master, F);

  nmod_poly_clear(m12);
  nmod_poly_clear(m21);
  nmod_poly_clear(m22);
  nmod_poly_clear(A);

  /* f = sgn * num / denom; normalize so that denom(0) = 1 */
  if (sgn < 0) {
    nmod_poly_scalar_mul_nmod(num, num, nmod_neg(1, mod));
  }
  ulong c = nmod_poly_get_coeff_ui(denom, 0);
  if (c == 0) {
    return 1;
  }
  ulong cinv = nmod_inv(c, mod);
  nmod_poly_scalar_mul_nmod(denom, denom, cinv);
  nmod_poly_scalar_mul_nmod(num, num, cinv);
  return 0;
}

int interpolate_univariate_ratfunc_from_data(const univariate_interp_data_t data,
					     const CF_p *vals,
					     nmod_poly_t num, nmod_poly_t denom) {

  nmod_t mod = data->master->mod;
  slong len = data->len;

  /* interpolating polynomial, written directly into the coefficients of F */
  nmod_poly_t F;
  nmod_poly_init_mod(F, mod);
  nmod_poly_fit_length(F, len);
  _nmod_poly_interpolate_nmod_vec_fast_precomp(F->coeffs, vals, data->tree,
                                               data->weights, len, mod);
  _nmod_poly_set_length(F, len);
  _nmod_poly_normalise(F); /* hgcd needs normalized input */

  int retcode = interpolate_univariate_ratfunc(F, data->master, num, denom);
  nmod_poly_clear(F);
  return retcode;
}

/* Search for a univariate rational function of unknown degree */

void ratfunc_search_init(ratfunc_search_t S, len_t ncoeffs, nmod_t mod,
                         flint_rand_t state) {

  S->ncoeffs = ncoeffs;

  S->lambda = malloc(ncoeffs * sizeof(CF_p));
  if (S->lambda == NULL) {
    fprintf(stderr, "Error when allocating for rational function search.\n");
    exit(1);
  }
  /* nonzero, so that every c_j enters the combination */
  for (len_t j = 0; j < ncoeffs; j++) {
    S->lambda[j] = 1 + n_randint(state, mod.n - 1);
  }

  nmod_poly_init_mod(S->F, mod);      /* F = 0: interpolates no points */
  nmod_poly_init_mod(S->master, mod);
  nmod_poly_one(S->master);           /* empty product */
  nmod_poly_init_mod(S->num, mod);
  nmod_poly_init_mod(S->den, mod);

  S->have_cand = 0;
  S->cand_pole_at_0 = 0;
  S->npts = 0;
  S->ncand = 0;
  return;
}

void ratfunc_search_clear(ratfunc_search_t S) {

  free(S->lambda);
  nmod_poly_clear(S->F);
  nmod_poly_clear(S->master);
  nmod_poly_clear(S->num);
  nmod_poly_clear(S->den);
  return;
}

int ratfunc_search_add_point(ratfunc_search_t S, CF_p t, const CF_p *vals) {

  nmod_t mod = S->F->mod;

  /* master(t) = prod (t - t_i) vanishes iff t is one of the t_i. Checked
   * first, so that S is not modified in that case. */
  CF_p master_at_t = nmod_poly_evaluate_nmod(S->master, t);
  if (master_at_t == 0) {
    return -1;
  }

  /* v = f(t), the value of the combination */
  CF_p v = 0;
  for (len_t j = 0; j < S->ncoeffs; j++) {
    v = nmod_add(v, nmod_mul(S->lambda[j], vals[j], mod), mod);
  }

  /* Test the candidate: it agrees with the point iff num(t) = v * den(t)
   * and den(t) != 0. Comparing num(t) with v * den(t) avoids an
   * inversion; den(t) = 0 means a pole of the candidate at t, but f(t) = v
   * is finite, so the candidate is wrong. */
  if (S->have_cand) {
    CF_p num_at_t = nmod_poly_evaluate_nmod(S->num, t);
    CF_p den_at_t = nmod_poly_evaluate_nmod(S->den, t);
    if (den_at_t == 0 || num_at_t != nmod_mul(v, den_at_t, mod)) {
      S->have_cand = 0;
    }
  }

  /* Newton step, also when the candidate agreed (if it fails later, the
   * next candidate needs all points):
   *   F <- F + (v - F(t)) / master(t) * master.
   * master vanishes at the old t_i, so the new F still takes the old
   * values there, and at t it takes the value v. */
  CF_p c = nmod_sub(v, nmod_poly_evaluate_nmod(S->F, t), mod);
  c = nmod_div(c, master_at_t, mod);
  nmod_poly_scalar_addmul_nmod(S->F, S->master, c);

  /* master <- (x - t) * master = x * master - t * master */
  nmod_poly_t tmp;
  nmod_poly_init_mod(tmp, mod);
  nmod_poly_shift_left(tmp, S->master, 1);
  nmod_poly_scalar_addmul_nmod(tmp, S->master, nmod_neg(t, mod));
  nmod_poly_swap(S->master, tmp);
  nmod_poly_clear(tmp);

  S->npts++;

  if (S->have_cand) {
    /* every point since the candidate was computed has agreed with it */
    return S->npts - S->ncand >= N_STABLE_POINTS_TO_STOP;
  }

  /* No candidate: compute one, but only for an odd number of points
   * N = 2d + 1, where hgcd gives numerator and denominator of degree <= d
   * (see interpolate_univariate_ratfunc). For even N, wait for the next
   * point. */
  if (S->npts % 2 == 1) {
    S->cand_pole_at_0 =
        interpolate_univariate_ratfunc(S->F, S->master, S->num, S->den);
    S->have_cand = 1;
    S->ncand = S->npts;
  }
  return 0;
}

int ratfunc_search_get(nmod_poly_t num, nmod_poly_t den,
                       const ratfunc_search_t S) {

  if (!S->have_cand) {
    return -1;
  }
  nmod_poly_set(num, S->num);
  nmod_poly_set(den, S->den);
  return S->cand_pole_at_0;
}
