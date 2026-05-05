#include "rsbench.h"

extern "C" {
void printF64(double x) { printf("%.4e\n", x); }
void printI64(int64_t x) { printf("%ld\n", x); }
}

int main() {
  Input input = {.nthreads = 1,
                 .n_nuclides = 355,
                 .lookups = 34,
                 .HM = LARGE,
                 .avg_n_poles = 1000,
                 .avg_n_windows = 100,
                 .numL = 4,
                 .doppler = 1,
                 .particles = 300000,
                 .simulation_method = HISTORY_BASED,
                 .kernel_id = 0};
  SimulationData GSD = initialize_simulation(input);

  double macro_xs[4] = {0};
  double d_macro_xs[4] = {1, 1, 1, 1};
  // calculate_macro_xs(macro_xs, /*mat=*/0, /*E=*/0.25, input, GSD.num_nucs,
  //                    GSD.mats, GSD.max_num_nucs, GSD.concs, GSD.n_windows,
  //                    GSD.pseudo_K0RS, GSD.windows, GSD.poles,
  //                    GSD.max_num_windows, GSD.max_num_poles);

  dcalculate_macro_xs(macro_xs, d_macro_xs, /*mat=*/0, /*E=*/0.25, input,
                      GSD.num_nucs, GSD.mats, GSD.max_num_nucs, GSD.concs,
                      GSD.n_windows, GSD.pseudo_K0RS, GSD.windows, GSD.poles,
                      GSD.d_poles, GSD.max_num_windows, GSD.max_num_poles);

  // Testing fast_nuclear_W
  // RSComplex Z = {.r = 137.372650, .i = -39.322330};
  // RSComplex primal = fast_nuclear_W(Z);
  // RSComplex adjoint = d_fast_nuclear_W(Z);
  // printf("primal result: %.5e %.5e\n", primal.r, primal.i);
  // printf("adjoint result: %.5e %.5e\n", adjoint.r, adjoint.i);

  int sz = 10;
  Pole *here = (Pole *)malloc(sz * sizeof(Pole));
  memcpy(here, &GSD.d_poles[64074], sz * sizeof(Pole));
  for (int i = 0; i < sz; ++i) {
    printf("here[%d]=%f %f %f %f %f %f %f %f\n", i, here[i].MP_EA.r,
           here[i].MP_EA.i, here[i].MP_RT.r, here[i].MP_RT.i, here[i].MP_RA.r,
           here[i].MP_RA.i, here[i].MP_RF.r, here[i].MP_RF.i);
  }
  free(here);

  printf("macro_xs: %.4e %.4e %.4e %.4e\n", macro_xs[0], macro_xs[1],
         macro_xs[2], macro_xs[3]);
}
