#ifndef MSOLVE_RATFUNC_INTERPOLATION_H
#define MSOLVE_RATFUNC_INTERPOLATION_H

#include <flint/nmod.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_mpoly.h>
#include "./msolve-data.h"
#include "../neogb/data.h"

typedef ulong CF_p; // for flint compatibility

void compute_newton_interpolant(const CF_p *vals, CF_p *const *grid, const deg_t *degrees,
				int32_t nvars,
				nmod_mpoly_t interpolant, const nmod_mpoly_ctx_t ctx);

#endif
