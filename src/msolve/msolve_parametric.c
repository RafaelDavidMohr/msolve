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

/* Main functions */

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

void compute_univariate_ratfunc_known_deg(const CF_p *args, const CF_p *vals,
                                          const deg_t deg,
                                          nmod_poly_t num, nmod_poly_t denom,
					  CF_p chr) {

  ulong N = 2*deg + 1;

  nmod_poly_t F, master;
  nmod_poly_init(F, chr);
  nmod_poly_init(master, chr);
  nmod_poly_interpolate_nmod_vec(F, args, vals, N);
  nmod_poly_product_roots_nmod_vec(master, args, N);

  nmod_poly_t m12, m21, m22, A;
  nmod_poly_init(m12, chr);
  nmod_poly_init(m21, chr);
  nmod_poly_init(m22, chr);
  nmod_poly_init(A, chr);

  slong sgn;
  sgn = nmod_poly_hgcd(denom, m12, m21, m22, A, num, master, F);

  ulong c = nmod_poly_get_coeff_ui(denom, 0);
  ulong cinv = nmod_inv(c, F->mod);
  nmod_poly_scalar_mul_nmod(denom, denom, cinv);
  nmod_poly_scalar_mul_nmod(num, num, sgn > 0 ? cinv : nmod_neg(cinv, F->mod));

  nmod_poly_clear(F);
  nmod_poly_clear(master);
  nmod_poly_clear(m12);
  nmod_poly_clear(m21);
  nmod_poly_clear(m22);
  nmod_poly_clear(A);

  return;
}


