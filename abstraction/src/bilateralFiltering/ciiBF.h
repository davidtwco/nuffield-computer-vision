#ifndef _CII_BF_H_
#define _CII_BF_H_

#define EE_MAX_IM_RANGE 256

typedef unsigned char uchar;

// This function implements fast box bilateral filtering using
// Coasine Integral Images (CII).

// CII are described in the paper:
// Cosine integral images for fast spatial and range filtering
// Elhanan Elboher, Michael Werman
// ICIP 2011
//
//
//
// PARAMETERS
//
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

void ciiBF(uchar *data, float *dataf, float *dctc, float *II, float *W, int height, int width, int nc, int r);

/////////////    Inline Functions     ///////////////////////////////////////////

inline
void imgRectSum_0(float* RS, const uchar* I, float* II, float c, const int height, const int width, const int ri, const int rj) {

	int ri1 = ri + 1;
	int rj1 = rj + 1;
	int ri21 = 2 * ri + 1;
	int rj21 = 2 * rj + 1;

	const uchar *pi = I;
	float *pii = II;

	const uchar *piw = pi + width;
	float *pii_p = pii;

	// first row
	*pii++ = c * (float)(*pi++);
	while (pi < piw) {
		*pii++ = (*pii_p++) + c * (float)(*pi++);
	}

	// integral image
	const uchar *piend = I + height * width;
	while (pi < piend) {

		// first sum the current row
		piw += width;
		pii_p = pii;
		*pii++ = c * (float)(*pi++);
		while (pi < piw) {
			(*pii++) = (*pii_p++) + c * (float)(*pi++);
		}

		// now add the sum of the upper rectangle
		float *piiw = pii;
		pii -= width;
		float *pii_p1 = pii - width;
		while (pii < piiw) {
			(*pii++) += (*pii_p1++);
		}
	}

	// rectangle sum
	float *pres = RS + ri1 * width;
	float *pend = RS + (height - ri) * width;

	float *pii1, *pii2, *pii3, *pii4;

	pii1 = II + ri21 * width + rj21;
	pii2 = II + ri21 * width;
	pii3 = II + rj21;
	pii4 = II;

	while (pres < pend) {

		float *pe = pres + width - rj;
		pres += rj1;
		while (pres < pe) {
			(*pres++) = (*pii1++) - (*pii2++) - (*pii3++) + (*pii4++);
		}
		pres += rj;

		pii1 += rj21;
		pii2 += rj21;
		pii3 += rj21;
		pii4 += rj21;

	}

}

inline
void add_lut(float* RS, const uchar* I, float* II, float* lut, float *lut1, const int height, const int width, const int ri, const int rj) {

	int ri1 = ri + 1;
	int rj1 = rj + 1;
	int ri21 = 2 * ri + 1;
	int rj21 = 2 * rj + 1;

	const uchar *pi = I;
	float *pii = II;

	const uchar *piw = pi + width;
	float *pii_p = pii;

	// first row
	*pii++ = lut[*pi++];
	while (pi < piw) {
		*pii++ = (*pii_p++) + lut[*pi++];
	}

	// integral image
	const uchar *piend = I + height * width;
	while (pi < piend) {

		// first sum the current row
		piw += width;
		pii_p = pii;
		*pii++ = lut[*pi++];
		while (pi < piw) {
			(*pii++) = (*pii_p++) + lut[*pi++];
		}

		// now add the sum of the upper rectangle
		float *piiw = pii;
		pii -= width;
		float *pii_p1 = pii - width;
		while (pii < piiw) {
			(*pii++) += (*pii_p1++);
		}
	}

	// rectangle sum
	float *pres = RS + ri1 * width;
	float *pend = RS + (height - ri) * width;

	float *pii1, *pii2, *pii3, *pii4;

	pii1 = II + ri21 * width + rj21;
	pii2 = II + ri21 * width;
	pii3 = II + rj21;
	pii4 = II;

	pi = I + ri1 * width;

	while (pres < pend) {

		float *pe = pres + width - rj;
		pres += rj1;
		pi += rj1;
		while (pres < pe) {
			//(*pres++) = (*pii1++) - (*pii2++) - (*pii3++) + (*pii4++);
			(*pres++) += lut1[*pi++] * ((*pii1++) - (*pii2++) - (*pii3++) + (*pii4++));
		}
		pres += rj;
		pi += rj;

		pii1 += rj21;
		pii2 += rj21;
		pii3 += rj21;
		pii4 += rj21;

	}

}

inline
void add_f_lut(float* RS, const uchar* I, float* II, float* lut, float* lut1, const int height, const int width, const int ri,
		const int rj) {

	int ri1 = ri + 1;
	int rj1 = rj + 1;
	int ri21 = 2 * ri + 1;
	int rj21 = 2 * rj + 1;

	const uchar *pi = I;
	float *pii = II;

	const uchar *piw = pi + width;
	float *pii_p = pii;

	// first row
	*pii++ = (float)(*pi) * lut[*pi];
	++pi;
	while (pi < piw) {
		*pii++ = (*pii_p++) + (float)(*pi) * lut[*pi];
		++pi;
	}

	// integral image
	const uchar *piend = I + height * width;
	while (pi < piend) {

		// first sum the current row
		piw += width;
		pii_p = pii;
		*pii++ = (float)(*pi) * lut[*pi];
		++pi;
		//*pii++ = (*pi++) * lut[*pi];
		while (pi < piw) {
			(*pii++) = (*pii_p++) + (float)(*pi) * lut[*pi];
			++pi;
			//(*pii++) = (*pii_p++) + (*pi++) * lut[*pi];
		}

		// now add the sum of the upper rectangle
		float *piiw = pii;
		pii -= width;
		float *pii_p1 = pii - width;
		while (pii < piiw) {
			(*pii++) += (*pii_p1++);
		}
	}

	// rectangle sum
	float *pres = RS + ri1 * width;
	float *pend = RS + (height - ri) * width;

	float *pii1, *pii2, *pii3, *pii4;

	pii1 = II + ri21 * width + rj21;
	pii2 = II + ri21 * width;
	pii3 = II + rj21;
	pii4 = II;

	pi = I + ri1 * width;

	while (pres < pend) {

		float *pe = pres + width - rj;
		pres += rj1;
		pi += rj1;
		while (pres < pe) {
			//(*pres++) = (*pii1++) - (*pii2++) - (*pii3++) + (*pii4++);
			(*pres++) += lut1[*pi++] * ((*pii1++) - (*pii2++) - (*pii3++) + (*pii4++));
		}
		pres += rj;
		pi += rj;

		pii1 += rj21;
		pii2 += rj21;
		pii3 += rj21;
		pii4 += rj21;

	}

}

#endif // _CII_BF_H_

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

