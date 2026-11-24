#include "./ratfunc_interpolation.h"

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

