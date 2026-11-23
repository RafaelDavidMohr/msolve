#include "./ratfunc_interpolation.h"

void compute_newton_interpolant(CF_p *vals, CF_p **grid, deg_t *degrees,
				int32_t nvars,
                                nmod_mpoly_t res, nmod_mpoly_ctx_t ctx) {

  if (nvars == 1) {

    nmod_poly_t rres;
    nmod_poly_init(rres, ctx->mod.n);
    nmod_poly_interpolate_nmod_vec(rres, vals, *grid, degrees[0]);
    nmod_poly_clear(rres);
    nmod_mpoly_set_nmod_poly(res, rres, 0, ctx);
    return;

  }

  nmod_mpoly_t master;
  nmod_mpoly_init(master, ctx);
  nmod_mpoly_set_ui(master, 1, ctx);
  CF_p* master_point;
  master_point = malloc(nvars * sizeof(CF_p));
  for (int i = 1; i < nvars; i++) {
    master_point[i] = 0;
  }

  deg_t degprod = 1;
  for (int i = 1; i < nvars; i++) {
    degprod *= (degrees[i] + 1);
  }

  nmod_mpoly_t res_from_next_var, summand;
  nmod_mpoly_init(res_from_next_var, ctx);
  nmod_mpoly_init(summand, ctx);

  CF_p mval;
  int val_ind = 0;
  for (int v = 0; v < degrees[0]; v++) {
    compute_newton_interpolant(&(vals[val_ind]), &(grid[1]),
			       &(degrees[1]), nvars-1,
			       res_from_next_var, ctx);
    val_ind += degprod;

    nmod_mpoly_sub_ui(summand, res_from_next_var, grid[0][v], ctx);
    nmod_mpoly_neg(summand, summand, ctx);

    // m(z_1) * (y_{r + 1} - f(y_{r + 1}, z2,...,zn)) / m(y_{r + 1})
    master_point[0] = grid[0][v];
    mval = nmod_mpoly_evaluate_all_ui(master, master_point, ctx);
    nmod_mpoly_scalar_mul_ui(summand, summand, nmod_inv(mval, ctx->mod), ctx);
    nmod_mpoly_mul(summand, summand, master, ctx);

    // add to current f
    nmod_mpoly_add(res, res, summand, ctx);
  }

  return;
}

