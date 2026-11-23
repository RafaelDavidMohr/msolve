#include <flint/nmod.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_mpoly.h>
#include "./msolve-data.h"
#include "../neogb/data.h"

typedef unsigned long CF_p; // for flint compatibility

void compute_newton_interpolant(CF_p *vals, CF_p **grid, deg_t *degrees,
				int32_t nvars,
                                nmod_mpoly_t res, nmod_mpoly_ctx_t ctx);
