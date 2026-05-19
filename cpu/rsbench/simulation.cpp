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
  for (int j = 0; j < 4; j++) {
    macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
  }
}

void calculate_sig_T(int nuc, double E, Input input, double *pseudo_K0RS,
                     RSComplex *sigTfactors) {
  double phi;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    phi = pseudo_K0RS[nuc * input.numL + i] * sqrt(E);
    if (i == 1)
      phi -= -atan(phi);
    else if (i == 2)
      phi -= atan(3.0 * phi / (3.0 - phi * phi));
    else if (i == 3)
      phi -= atan(phi * (15.0 - phi * phi) / (15.0 - 6.0 * phi * phi));

    phi *= 2.0;

    sigTfactors[i].r = +cos(phi);
    sigTfactors[i].i = -sin(phi);
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
    RSComplex prefactor = {0, 8.124330e+01};
    double an[10] = {2.758402e-01, 2.245740e-01, 1.594149e-01, 9.866577e-02,
                     5.324414e-02, 2.505215e-02, 1.027747e-02, 3.676164e-03,
                     1.146494e-03, 3.117570e-04};
    double neg_1n[10] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};

    double denominator_left[10] = {
        9.869604e+00, 3.947842e+01, 8.882644e+01, 1.579137e+02, 2.467401e+02,
        3.553058e+02, 4.836106e+02, 6.316547e+02, 7.994380e+02, 9.869604e+02};

    RSComplex t1 = {0, 12};
    RSComplex t2 = {12, 0};
    RSComplex i = {0, 1};
    RSComplex one = {1, 0};
    RSComplex W =
        c_div(c_mul(i, (c_sub(one, fast_cexp(c_mul(t1, Z))))), c_mul(t2, Z));
    RSComplex sum = {0, 0};
#pragma unroll
    for (int n = 0; n < 10; n++) {
      RSComplex t3 = {neg_1n[n], 0};
      RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
      RSComplex t4 = {denominator_left[n], 0};
      RSComplex t5 = {144, 0};
      RSComplex bot = c_sub(t4, c_mul(t5, c_mul(Z, Z)));
      RSComplex t6 = {an[n], 0};
      sum = c_add(sum, c_mul(t6, c_div(top, bot)));
    }
    W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
    return W;
  } else {
    // QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).
    // Pre-computed parameters
    RSComplex a = {
        0.512424224754768462984202823134979415014943561548661637413182, 0};
    RSComplex b = {
        0.275255128608410950901357962647054304017026259671664935783653, 0};
    RSComplex c = {
        0.051765358792987823963876628425793170829107067780337219430904, 0};
    RSComplex d = {
        2.724744871391589049098642037352945695982973740328335064216346, 0};

    RSComplex i = {0, 1};
    RSComplex Z2 = c_mul(Z, Z);
    // Three Term Asymptotic Expansion
    RSComplex W =
        c_mul(c_mul(Z, i),
              (c_add(c_div(a, (c_sub(Z2, b))), c_div(c, (c_sub(Z2, d))))));

    return W;
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
  sigA = E * w.A;
  sigF = E * w.F;

  double dopp = 0.5;

  // if (w.start == 0)
  //	printf("start=%d\n", w.start);
  //  Loop over Poles within window, add contributions
  int i = w.start + 1;
  // nuc was 58
  Pole pole = poles[nuc * max_num_poles + i];
  // printf("here: %d\n",  nuc);

  // Prep Z
  RSComplex Z = pole.MP_EA;

  // Evaluate Fadeeva Function
  RSComplex faddeeva = fast_nuclear_W(Z);

  // Update W
  sigT += faddeeva.r;

  sigE = sigT + sigA;

  micro_xs[0] = sigT;
  micro_xs[1] = sigA;
  micro_xs[2] = sigF;
  micro_xs[3] = sigE;
}
