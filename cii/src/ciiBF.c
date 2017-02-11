#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ciiBF.h"

// This function implements fast box bilateral filtering using
// Coasine Integral Images (CII).
//
// CII are described in the paper:
// Cosine integral images for fast spatial and range filtering
// Elhanan Elboher, Michael Werman
// ICIP 2011
// PARAMETERS
// data: pointer to a 1D float array which represents a
//       (height x width) grayscale image.
//       The (i,j) pixel should be placed at data[i*width+j].
//
// dataf: the filtering result, a pointer to 1D array of size height*width.
//
// dctc: array of DCT coefficients of the used Gaussian range kernel.
//
// II, W: two pointers to auxiliary 1D arrays, each of size height*width.
//
// height: column length (= number of image rows).
// 
// width: row length (= number of image columns).
//
// nc: number of DCT coefficients for kernel approximation, 0 ... (nc-1).
//
// r: spatial radius; the used kernel / window size is (2*r+1) x (2*r+1).

void ciiBF(uchar *data, float *dataf, float *dctc, float *II, float *W, int height, int width, int nc, int r) {

	// ----------------  memory allocation  --------------------------

	// lookup tables
	// (size depened on range, usually small constant * 256)
	float *cR = (float*) malloc((nc - 1) * EE_MAX_IM_RANGE * sizeof(float));
	float *sR = (float*) malloc((nc - 1) * EE_MAX_IM_RANGE * sizeof(float));
	float *dcR = (float*) malloc((nc - 1) * EE_MAX_IM_RANGE * sizeof(float));
	float *dsR = (float*) malloc((nc - 1) * EE_MAX_IM_RANGE * sizeof(float));

	// ----------------  input dependent setup  --------------------------

	int fx;
	int r2 = pow(2 * r + 1, 2);
	float c0 = dctc[0];
	float c0r2 = dctc[0] * r2;

	int ck, ckr;

	float tmpfx;

	for (ck = 1; ck < nc; ck++) {

		ckr = (ck - 1) * EE_MAX_IM_RANGE;

		for (fx = 0; fx < EE_MAX_IM_RANGE; fx++) {

			tmpfx = M_PI * (float)(fx) * (float)(ck) / EE_MAX_IM_RANGE;
			cR[ckr + fx] = cos(tmpfx);
			sR[ckr + fx] = sin(tmpfx);
			dcR[ckr + fx] = dctc[ck] * cos(tmpfx);
			dsR[ckr + fx] = dctc[ck] * sin(tmpfx);

		}

	}

	// ----------------------- filtering  --------------------------------

	// =======
	// weights
	// =======

	// initialize weights
	float *pw = W;
	float *pwe = W + height * width;
	while (pw < pwe) {
		*pw++ = c0r2;
	}

	for (ck = 1; ck < nc; ck++) {

		ckr = (ck - 1) * EE_MAX_IM_RANGE;

		// add cosine term to W
		add_lut(W, data, II, cR + ckr, dcR + ckr, height, width, r, r);

		// add sine term to W
		add_lut(W, data, II, sR + ckr, dsR + ckr, height, width, r, r);

	}

	// ==============
	// values (dataf)
	// ==============

	// initialize dataf
	imgRectSum_0(dataf, data, II, c0, height, width, r, r);

	for (ck = 1; ck < nc; ck++) {

		ckr = (ck - 1) * EE_MAX_IM_RANGE;

		// add cosine term to dataf
		add_f_lut(dataf, data, II, cR + ckr, dcR + ckr, height, width, r, r);

		// add sine term to dataf
		add_f_lut(dataf, data, II, sR + ckr, dsR + ckr, height, width, r, r);

	}

	// ======
	// divide
	// ======

	float *pd = dataf + r * width;
	pw = W + r * width;

	float *pdie = pd + width - r;
	float *pdend = dataf + (height - r) * width;

	while (pd < pdend) {
		pd += r;
		pw += r;
		while (pd < pdie) {
			(*pd++) /= ((*pw++) * EE_MAX_IM_RANGE);
		}
		pd += r;
		pw += r;
		pdie += width;
	}

	// free lookup tables
	free(cR);
	free(sR);
	free(dcR);
	free(dsR);

}

// Copyright (c) 2011, Elhanan Elboher
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met: 
//    * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//    * Neither the name of the The Hebrew University of Jerusalem nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

