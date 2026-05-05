#include "rsbench.h"

template <typename Ret, typename... Args>
Ret __enzyme_autodiff(void *, Args...);

extern int enzyme_dup, enzyme_out, enzyme_const;

void dcalculate_macro_xs(double *macro_xs, double *dmacro_xs, int mat, double E,
                         Input input, int *num_nucs, int *mats,
                         int max_num_nucs, double *concs, int *n_windows,
                         double *pseudo_K0Rs, Window *windows, Pole *poles,
                         Pole *dpoles, int max_num_windows, int max_num_poles) {
  // clang-format off
  __enzyme_autodiff<void>(
    (void*)calculate_macro_xs,
    enzyme_dup, macro_xs, dmacro_xs,
    enzyme_const, mat,
    enzyme_const, E,
    enzyme_const, input,
    enzyme_const, num_nucs,
    enzyme_const, mats,
    enzyme_const, max_num_nucs,
    enzyme_const, concs,
    enzyme_const, n_windows,
    enzyme_const, pseudo_K0Rs,
    enzyme_const, windows,
    enzyme_dup, poles, dpoles,
    enzyme_const, max_num_windows,
    enzyme_const, max_num_poles
  );
  // clang-format on
}

RSComplex c_add(RSComplex A, RSComplex B) {
  RSComplex C;
  C.r = A.r + B.r;
  C.i = A.i + B.i;
  return C;
}

RSComplex c_sub(RSComplex A, RSComplex B) {
  RSComplex C;
  C.r = A.r - B.r;
  C.i = A.i - B.i;
  return C;
}

RSComplex c_mul(RSComplex A, RSComplex B) {
  double a = A.r;
  double b = A.i;
  double c = B.r;
  double d = B.i;
  RSComplex C;
  C.r = (a * c) - (b * d);
  C.i = (a * d) + (b * c);
  return C;
}

RSComplex c_div(RSComplex A, RSComplex B) {
  double a = A.r;
  double b = A.i;
  double c = B.r;
  double d = B.i;
  RSComplex C;
  double denom = c * c + d * d;
  C.r = ((a * c) + (b * d)) / denom;
  C.i = ((b * c) - (a * d)) / denom;
  return C;
}

double c_abs(RSComplex A) { return sqrt(A.r * A.r + A.i * A.i); }

// Fast (but inaccurate) exponential function
// Written By "ACMer":
// https://codingforspeed.com/using-faster-exponential-approximation/
// We use our own to avoid small differences in compiler specific
// exp() intrinsic implementations that make it difficult to verify
// if the code is working correctly or not.
double fast_exp(double x) {
  x = 1.0 + x * 0.000244140625;
  // clang-format off
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  // clang-format on
  return x;
}

// Implementation based on:
// z = x + iy
// cexp(z) = e^x * (cos(y) + i * sin(y))
RSComplex fast_cexp(RSComplex z) {
  double x = z.r;
  double y = z.i;

  // For consistency across architectures, we
  // will use our own exponetial implementation
  // double t1 = exp(x);
  double t1 = fast_exp(x);
  double t2 = cos(y);
  double t3 = sin(y);
  RSComplex t4 = {t2, t3};
  RSComplex t5 = {t1, 0};
  RSComplex result = c_mul(t5, (t4));
  return result;
}

void calculate_macro_xs(double *__restrict__ macro_xs, int mat, double E,
                        Input input, int *__restrict__ num_nucs,
                        int *__restrict__ mats, int max_num_nucs,
                        double *__restrict__ concs, int *__restrict__ n_windows,
                        double *__restrict__ pseudo_K0Rs,
                        Window *__restrict__ windows, Pole *__restrict__ poles,
                        int max_num_windows, int max_num_poles) {
  // zero out macro vector
  for (int i = 0; i < 4; i++)
    macro_xs[i] = 0;

  // for nuclide in mat
  int sz = num_nucs[mat];
  int i = 0;
  double micro_xs[4] = {0};
  int nuc = mats[mat * max_num_nucs + i];

  calculate_micro_xs_doppler(micro_xs, nuc, E, input, n_windows, pseudo_K0Rs,
                             windows, poles, max_num_windows, max_num_poles);
  for (int j = 1; j < 3; j++) {
    macro_xs[j] += micro_xs[j];
  }
}

void calculate_sig_T(int nuc, double E, Input input, double *pseudo_K0RS,
                     RSComplex *sigTfactors) {
  double phi;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    phi = 0;

    sigTfactors[i].r = (phi);
    sigTfactors[i].i = (phi);
  }
}

void fast_nuclear_W_wrapper(RSComplex *Z, RSComplex *out) {
  *out = fast_nuclear_W(*Z);
}

RSComplex d_fast_nuclear_W(RSComplex Z) {
  RSComplex out, dZ = {0, 0};
  RSComplex dout = {1.0, 1.0};
  // __enzyme_autodiff<void>((void *)fast_nuclear_W_wrapper, enzyme_dup, &Z,
  // &dZ,
  //                         enzyme_dup, &out, &dout);
  return dZ;
}

__attribute__((always_inline)) RSComplex fast_nuclear_W(RSComplex Z) {
  // Abrarov
  if (c_abs(Z) < 6.0) {
    // Precomputed parts for speeding things up
    // (N = 10, Tm = 12.0)

    RSComplex one = {1, 0};
    RSComplex sum = {0, 0};
#pragma unroll
    for (int n = 0; n < 8; n++) {
      RSComplex t3 = {(n & 1) ? 1.0 : -1.0, 0};
      RSComplex top = c_sub(c_mul(t3, fast_cexp(Z)), one);
      RSComplex bot = c_mul(Z, Z);
      sum = c_add(sum, c_div(top, bot));
    }
    // W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
    RSComplex W = c_mul(Z, sum);
    return W;
  } else {
    // QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).
    // Pre-computed parameters
    // RSComplex a = {
    //     0.512424224754768462984202823134979415014943561548661637413182, 0};
    // RSComplex b = {
    //     0.275255128608410950901357962647054304017026259671664935783653, 0};
    // RSComplex c = {
    //     0.051765358792987823963876628425793170829107067780337219430904, 0};
    // RSComplex d = {
    //     2.724744871391589049098642037352945695982973740328335064216346, 0};

    // RSComplex i = {0, 1};
    // RSComplex Z2 = c_mul(Z, Z);
    // // Three Term Asymptotic Expansion
    // RSComplex W =
    //     c_mul(c_mul(Z, i),
    //           (c_add(c_div(a, (c_sub(Z2, b))), c_div(c, (c_sub(Z2, d))))));

    // return W;
    return Z;
  }
}

inline void calculate_micro_xs_doppler(double *micro_xs, int nuc, double E,
                                       Input input, int *n_windows,
                                       double *pseudo_K0RS, Window *windows,
                                       Pole *poles, int max_num_windows,
                                       int max_num_poles) {
  // MicroScopic XS's to Calculate
  double sigT;
  double sigA;
  double sigF;
  double sigE;

  // Calculate Window Index
  double spacing = 1.0 / n_windows[nuc];
  int window = (int)(E / spacing);
  if (window == n_windows[nuc])
    window--;
  // int window = 23;

  // Calculate sigTfactors
  RSComplex sigTfactors[4] = {
      {0, 0},
      {0, 0},
      {0, 0},
      {0, 0}}; // Of length input.numL, which is always 4
  // calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors);

  // Calculate contributions from window "background" (i.e., poles outside
  // window (pre-calculated)
  Window w = windows[nuc * max_num_windows + window];
  sigT = E * w.T;
  sigA = w.A;
  sigF = E * w.F;

  double dopp = 0.5;

  // if (w.start == 0)
  //	printf("start=%d\n", w.start);
  //  Loop over Poles within window, add contributions
  int i = w.start;
  // for (int i = w.start; i < w.end; i++) {
  // nuc was 58
  Pole pole = poles[nuc * max_num_poles + i];
  // printf("here: %d\n",  nuc);

  // Prep Z
  RSComplex Z = pole.MP_EA;
  // RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);

  // Evaluate Fadeeva Function
  RSComplex faddeeva = fast_nuclear_W(Z);

  // Update W
  sigT += faddeeva.r;
  // sigT += (c_mul(pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]))).r;
  // sigA += (c_mul(pole.MP_RA, faddeeva)).r;
  // sigF += (c_mul(pole.MP_RF, faddeeva)).r;
  // }

  // sigE = sigT - sigA;
  sigE = sigT + sigA;

  micro_xs[0] = 0.0;
  micro_xs[1] = 0.0;
  micro_xs[2] = sigF;
  micro_xs[3] = sigE;
  // micro_xs[1] = sigA;
  // micro_xs[2] = sigF;
  // micro_xs[3] = sigE;
}
