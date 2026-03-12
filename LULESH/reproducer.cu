//   clang++ -DNDEBUG -ffast-math --cuda-path=/path/to/cuda/ \
//     -std=c++11 -fno-exceptions --cuda-gpu-arch=sm_70 \
//     --no-cuda-version-check \
//     -fplugin=/path/to/ClangEnzyme-22.so \
//     -Xclang -add-plugin -Xclang enzyme \
//     -mllvm -raising-plugin-path -mllvm /path/to/libRaise.so \
//     -mllvm -reactant-backend -mllvm cuda \
//     -x cuda reproducer.cu -c -o /dev/null
//
// ISL emits "unexpected unnamed parameters" errors,
// then "Reactant: failed to run mlir passes".
// On full LULESH, this leads to OOM-kill.

typedef double Real_t;
typedef int    Index_t;

__device__ inline Real_t CBRT(Real_t arg) { return cbrt(arg); }

// Volume derivative for one node permutation (18-arg cross product)
__device__ __forceinline__
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0);
   *dvdx = ((y1+y2)*(z0+z1) - (y0+y1)*(z1+z2) + (y0+y4)*(z3+z4)
          - (y3+y4)*(z0+z4) - (y2+y5)*(z3+z5) + (y3+y5)*(z2+z5)) * twelfth;
   *dvdy = (-(x1+x2)*(z0+z1) + (x0+x1)*(z1+z2) - (x0+x4)*(z3+z4)
          + (x3+x4)*(z0+z4) + (x2+x5)*(z3+z5) - (x3+x5)*(z2+z5)) * twelfth;
   *dvdz = (-(y1+y2)*(x0+x1) + (y0+y1)*(x1+x2) - (y0+y4)*(x3+x4)
          + (y3+y4)*(x0+x4) + (y2+y5)*(x3+x5) - (y3+y5)*(x2+x5)) * twelfth;
}

// 8 calls to VoluDer with permuted node indices
__device__ __forceinline__
void CalcElemVolumeDerivative(Real_t dvdx[8], Real_t dvdy[8], Real_t dvdz[8],
                              const Real_t x[8], const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1],x[2],x[3],x[4],x[5],x[7], y[1],y[2],y[3],y[4],y[5],y[7],
           z[1],z[2],z[3],z[4],z[5],z[7], &dvdx[0],&dvdy[0],&dvdz[0]);
   VoluDer(x[0],x[1],x[2],x[7],x[4],x[6], y[0],y[1],y[2],y[7],y[4],y[6],
           z[0],z[1],z[2],z[7],z[4],z[6], &dvdx[3],&dvdy[3],&dvdz[3]);
   VoluDer(x[3],x[0],x[1],x[6],x[7],x[5], y[3],y[0],y[1],y[6],y[7],y[5],
           z[3],z[0],z[1],z[6],z[7],z[5], &dvdx[2],&dvdy[2],&dvdz[2]);
   VoluDer(x[2],x[3],x[0],x[5],x[6],x[4], y[2],y[3],y[0],y[5],y[6],y[4],
           z[2],z[3],z[0],z[5],z[6],z[4], &dvdx[1],&dvdy[1],&dvdz[1]);
   VoluDer(x[7],x[6],x[5],x[0],x[3],x[1], y[7],y[6],y[5],y[0],y[3],y[1],
           z[7],z[6],z[5],z[0],z[3],z[1], &dvdx[4],&dvdy[4],&dvdz[4]);
   VoluDer(x[4],x[7],x[6],x[1],x[0],x[2], y[4],y[7],y[6],y[1],y[0],y[2],
           z[4],z[7],z[6],z[1],z[0],z[2], &dvdx[5],&dvdy[5],&dvdz[5]);
   VoluDer(x[5],x[4],x[7],x[2],x[1],x[3], y[5],y[4],y[7],y[2],y[1],y[3],
           z[5],z[4],z[7],z[2],z[1],z[3], &dvdx[6],&dvdy[6],&dvdz[6]);
   VoluDer(x[6],x[5],x[4],x[3],x[2],x[0], y[6],y[5],y[4],y[3],y[2],y[0],
           z[6],z[5],z[4],z[3],z[2],z[0], &dvdx[7],&dvdy[7],&dvdz[7]);
}

// Jacobian + cofactor matrix for hexahedral element
__device__ __forceinline__
void CalcElemShapeFunctionDerivatives(const Real_t* const x,
                                      const Real_t* const y,
                                      const Real_t* const z,
                                      Real_t b[][8], Real_t* const volume)
{
  const Real_t x0=x[0], x1=x[1], x2=x[2], x3=x[3],
               x4=x[4], x5=x[5], x6=x[6], x7=x[7];
  const Real_t y0=y[0], y1=y[1], y2=y[2], y3=y[3],
               y4=y[4], y5=y[5], y6=y[6], y7=y[7];
  const Real_t z0=z[0], z1=z[1], z2=z[2], z3=z[3],
               z4=z[4], z5=z[5], z6=z[6], z7=z[7];
  Real_t fjxxi = .125*((x6-x0)+(x5-x3)-(x7-x1)-(x4-x2));
  Real_t fjxet = .125*((x6-x0)-(x5-x3)+(x7-x1)-(x4-x2));
  Real_t fjxze = .125*((x6-x0)+(x5-x3)+(x7-x1)+(x4-x2));
  Real_t fjyxi = .125*((y6-y0)+(y5-y3)-(y7-y1)-(y4-y2));
  Real_t fjyet = .125*((y6-y0)-(y5-y3)+(y7-y1)-(y4-y2));
  Real_t fjyze = .125*((y6-y0)+(y5-y3)+(y7-y1)+(y4-y2));
  Real_t fjzxi = .125*((z6-z0)+(z5-z3)-(z7-z1)-(z4-z2));
  Real_t fjzet = .125*((z6-z0)-(z5-z3)+(z7-z1)-(z4-z2));
  Real_t fjzze = .125*((z6-z0)+(z5-z3)+(z7-z1)+(z4-z2));
  Real_t cjxxi =  (fjyet*fjzze)-(fjzet*fjyze);
  Real_t cjxet = -(fjyxi*fjzze)+(fjzxi*fjyze);
  Real_t cjxze =  (fjyxi*fjzet)-(fjzxi*fjyet);
  Real_t cjyxi = -(fjxet*fjzze)+(fjzet*fjxze);
  Real_t cjyet =  (fjxxi*fjzze)-(fjzxi*fjxze);
  Real_t cjyze = -(fjxxi*fjzet)+(fjzxi*fjxet);
  Real_t cjzxi =  (fjxet*fjyze)-(fjyet*fjxze);
  Real_t cjzet = -(fjxxi*fjyze)+(fjyxi*fjxze);
  Real_t cjzze =  (fjxxi*fjyet)-(fjyxi*fjxet);
  b[0][0]=-cjxxi-cjxet-cjxze; b[0][1]= cjxxi-cjxet-cjxze;
  b[0][2]= cjxxi+cjxet-cjxze; b[0][3]=-cjxxi+cjxet-cjxze;
  b[0][4]=-b[0][2]; b[0][5]=-b[0][3]; b[0][6]=-b[0][0]; b[0][7]=-b[0][1];
  b[1][0]=-cjyxi-cjyet-cjyze; b[1][1]= cjyxi-cjyet-cjyze;
  b[1][2]= cjyxi+cjyet-cjyze; b[1][3]=-cjyxi+cjyet-cjyze;
  b[1][4]=-b[1][2]; b[1][5]=-b[1][3]; b[1][6]=-b[1][0]; b[1][7]=-b[1][1];
  b[2][0]=-cjzxi-cjzet-cjzze; b[2][1]= cjzxi-cjzet-cjzze;
  b[2][2]= cjzxi+cjzet-cjzze; b[2][3]=-cjzxi+cjzet-cjzze;
  b[2][4]=-b[2][2]; b[2][5]=-b[2][3]; b[2][6]=-b[2][0]; b[2][7]=-b[2][1];
  *volume = 8.*(fjxet*cjxet + fjyet*cjyet + fjzet*cjzet);
}

// Cross-product face normal, accumulated to 4 nodes
__device__ __forceinline__
void SumElemFaceNormal(Real_t *nx0, Real_t *ny0, Real_t *nz0,
                       Real_t *nx1, Real_t *ny1, Real_t *nz1,
                       Real_t *nx2, Real_t *ny2, Real_t *nz2,
                       Real_t *nx3, Real_t *ny3, Real_t *nz3,
                       Real_t x0, Real_t y0, Real_t z0,
                       Real_t x1, Real_t y1, Real_t z1,
                       Real_t x2, Real_t y2, Real_t z2,
                       Real_t x3, Real_t y3, Real_t z3)
{
   Real_t bx0=.5*(x3+x2-x1-x0), by0=.5*(y3+y2-y1-y0), bz0=.5*(z3+z2-z1-z0);
   Real_t bx1=.5*(x2+x1-x3-x0), by1=.5*(y2+y1-y3-y0), bz1=.5*(z2+z1-z3-z0);
   Real_t ax=.25*(by0*bz1-bz0*by1);
   Real_t ay=.25*(bz0*bx1-bx0*bz1);
   Real_t az=.25*(bx0*by1-by0*bx1);
   *nx0+=ax; *nx1+=ax; *nx2+=ax; *nx3+=ax;
   *ny0+=ay; *ny1+=ay; *ny2+=ay; *ny3+=ay;
   *nz0+=az; *nz1+=az; *nz2+=az; *nz3+=az;
}

// 6 face normals for hex element
__device__ __forceinline__
void CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8], Real_t pfz[8],
                         const Real_t x[8], const Real_t y[8],
                         const Real_t z[8])
{
   for (int i=0;i<8;++i) { pfx[i]=0.; pfy[i]=0.; pfz[i]=0.; }
   SumElemFaceNormal(&pfx[0],&pfy[0],&pfz[0],&pfx[1],&pfy[1],&pfz[1],
     &pfx[2],&pfy[2],&pfz[2],&pfx[3],&pfy[3],&pfz[3],
     x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2],x[3],y[3],z[3]);
   SumElemFaceNormal(&pfx[0],&pfy[0],&pfz[0],&pfx[4],&pfy[4],&pfz[4],
     &pfx[5],&pfy[5],&pfz[5],&pfx[1],&pfy[1],&pfz[1],
     x[0],y[0],z[0],x[4],y[4],z[4],x[5],y[5],z[5],x[1],y[1],z[1]);
   SumElemFaceNormal(&pfx[1],&pfy[1],&pfz[1],&pfx[5],&pfy[5],&pfz[5],
     &pfx[6],&pfy[6],&pfz[6],&pfx[2],&pfy[2],&pfz[2],
     x[1],y[1],z[1],x[5],y[5],z[5],x[6],y[6],z[6],x[2],y[2],z[2]);
   SumElemFaceNormal(&pfx[2],&pfy[2],&pfz[2],&pfx[6],&pfy[6],&pfz[6],
     &pfx[7],&pfy[7],&pfz[7],&pfx[3],&pfy[3],&pfz[3],
     x[2],y[2],z[2],x[6],y[6],z[6],x[7],y[7],z[7],x[3],y[3],z[3]);
   SumElemFaceNormal(&pfx[3],&pfy[3],&pfz[3],&pfx[7],&pfy[7],&pfz[7],
     &pfx[4],&pfy[4],&pfz[4],&pfx[0],&pfy[0],&pfz[0],
     x[3],y[3],z[3],x[7],y[7],z[7],x[4],y[4],z[4],x[0],y[0],z[0]);
   SumElemFaceNormal(&pfx[4],&pfy[4],&pfz[4],&pfx[7],&pfy[7],&pfz[7],
     &pfx[6],&pfy[6],&pfz[6],&pfx[5],&pfy[5],&pfz[5],
     x[4],y[4],z[4],x[7],y[7],z[7],x[6],y[6],z[6],x[5],y[5],z[5]);
}

// 8x4 hourglass mode matrix (32 assignments, each with 8-node sums)
__device__ __forceinline__
void CalcHourglassModes(const Real_t xn[8], const Real_t yn[8],
                        const Real_t zn[8],
                        const Real_t dvdx[8], const Real_t dvdy[8],
                        const Real_t dvdz[8],
                        Real_t hg[8][4], Real_t vi)
{
    Real_t hx, hy, hz;
    // Each mode: compute 3 weighted sums of 8 nodes, then 8 hourglass entries
    #define MODE(M, s0,s1,s2,s3,s4,s5,s6,s7) \
      hx=s0*xn[0]+s1*xn[1]+s2*xn[2]+s3*xn[3]+s4*xn[4]+s5*xn[5]+s6*xn[6]+s7*xn[7]; \
      hy=s0*yn[0]+s1*yn[1]+s2*yn[2]+s3*yn[3]+s4*yn[4]+s5*yn[5]+s6*yn[6]+s7*yn[7]; \
      hz=s0*zn[0]+s1*zn[1]+s2*zn[2]+s3*zn[3]+s4*zn[4]+s5*zn[5]+s6*zn[6]+s7*zn[7]; \
      for(int i=0;i<8;i++) hg[i][M]=Real_t(s0)-vi*(dvdx[i]*hx+dvdy[i]*hy+dvdz[i]*hz);
    // Note: the s0..s7 signs vary per mode; using exact LULESH sign patterns
    MODE(0, 1, 1,-1,-1,-1,-1, 1, 1)
    MODE(1, 1,-1,-1, 1,-1, 1, 1,-1)
    MODE(2, 1,-1, 1,-1, 1,-1, 1,-1)
    MODE(3,-1, 1,-1, 1, 1,-1, 1,-1)
    #undef MODE
}

// 8x4 matrix-vector product done 3x (x,y,z) = 96 unrolled mul-acc ops
__device__ __forceinline__
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                              Real_t hg[8][4], Real_t coeff,
                              Real_t *fx, Real_t *fy, Real_t *fz)
{
   // For each coordinate: h = hg^T * vel (4 dots of len 8),
   //                      force += coeff * hg * h (8 dots of len 4)
   #define DO_COORD(VEL, FORCE) { \
     Real_t h0=0,h1=0,h2=0,h3=0; \
     for(int i=0;i<8;i++){h0+=hg[i][0]*VEL[i];h1+=hg[i][1]*VEL[i]; \
                           h2+=hg[i][2]*VEL[i];h3+=hg[i][3]*VEL[i];} \
     for(int i=0;i<8;i++) \
       FORCE[i]+=coeff*(hg[i][0]*h0+hg[i][1]*h1+hg[i][2]*h2+hg[i][3]*h3); \
   }
   DO_COORD(xd, fx)
   DO_COORD(yd, fy)
   DO_COORD(zd, fz)
   #undef DO_COORD
}

// =========================================================================
// THE KERNEL — after inlining all __forceinline__ functions above, this
// becomes one massive basic block with ~500+ arithmetic operations and
// scatter-gather memory accesses that cause ISL to OOM.
// =========================================================================
__global__ __launch_bounds__(64, 4)
void CalcVolumeForceForElems_kernel(
    const Real_t* __restrict__ volo,
    const Real_t* __restrict__ v,
    const Real_t* __restrict__ p,
    const Real_t* __restrict__ q,
    Real_t hourg,
    Index_t padded_numElem,
    const Index_t* __restrict__ nodelist,
    const Real_t* __restrict__ ss,
    const Real_t* __restrict__ elemMass,
    const Real_t* __restrict__ x,
    const Real_t* __restrict__ y,
    const Real_t* __restrict__ z,
    const Real_t* __restrict__ xd,
    const Real_t* __restrict__ yd,
    const Real_t* __restrict__ zd,
    Real_t* __restrict__ fx_elem,
    Real_t* __restrict__ fy_elem,
    Real_t* __restrict__ fz_elem,
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads)
{
    int elem = blockDim.x * blockIdx.x + threadIdx.x;
    if (elem >= num_threads) return;

    Real_t xn[8], yn[8], zn[8], xdn[8], ydn[8], zdn[8];
    Real_t dvdxn[8], dvdyn[8], dvdzn[8];
    Real_t hgfx[8], hgfy[8], hgfz[8];
    Real_t hourgam[8][4];

    Real_t det = volo[elem] * v[elem];
    Real_t sigxx = -p[elem] - q[elem];

    // Scatter-gather: indirect index through nodelist
    Index_t n[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) n[i] = nodelist[elem + i*padded_numElem];

    Real_t volinv = 1.0 / det;

    #pragma unroll
    for (int i = 0; i < 8; i++) { xn[i]=x[n[i]]; yn[i]=y[n[i]]; zn[i]=z[n[i]]; }

    Real_t coeff = -hourg * 0.01 * ss[elem] * elemMass[elem] / CBRT(det);

    CalcElemVolumeDerivative(dvdxn, dvdyn, dvdzn, xn, yn, zn);
    CalcHourglassModes(xn, yn, zn, dvdxn, dvdyn, dvdzn, hourgam, volinv);

    Real_t B[3][8];
    CalcElemShapeFunctionDerivatives(xn, yn, zn, B, &det);
    CalcElemNodeNormals(B[0], B[1], B[2], xn, yn, zn);

    if (det < 0.) *bad_vol = elem;

    #pragma unroll
    for (int i=0;i<8;i++) {
        hgfx[i]=-(sigxx*B[0][i]); hgfy[i]=-(sigxx*B[1][i]); hgfz[i]=-(sigxx*B[2][i]);
    }

    #pragma unroll
    for (int i=0;i<8;i++) { xdn[i]=xd[n[i]]; ydn[i]=yd[n[i]]; zdn[i]=zd[n[i]]; }

    CalcElemFBHourglassForce(xdn, ydn, zdn, hourgam, coeff, hgfx, hgfy, hgfz);

    #pragma unroll
    for (int i=0;i<8;i++) {
        Index_t loc = elem + padded_numElem*i;
        fx_elem[loc]=hgfx[i]; fy_elem[loc]=hgfy[i]; fz_elem[loc]=hgfz[i];
    }
}

int main() {
    CalcVolumeForceForElems_kernel<<<1,64>>>(
        nullptr,nullptr,nullptr,nullptr,0.,0,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0);
    cudaDeviceSynchronize();
}
