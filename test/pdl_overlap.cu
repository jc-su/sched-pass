// pdl_overlap.cu -- validation fixture for the PDL lever (SCHED_PDL=1).
//
// Two dependent kernels on one stream: producer writes buf, consumer (woven:
// pi remap + timer + PDL points) reads it. The weave puts griddepcontrol.wait
// AFTER the consumer's control-table reads, so with the launch attribute
// (cudaLaunchAttributeProgrammaticStreamSerialization) those PCIe reads
// overlap the producer's tail; launch_dependents at the producer's returns
// releases the consumer early.
//
// Gates (E0 discipline: a scheduling hint must never change results):
//   * consumer output bit-exact:  PDL launch == plain launch
//   * woven timer still populates under PDL
// Reported (not gated -- idle-GPU timing is noisy): paired-launch wall time
// with vs without the attribute.
//
// Build (plugin, PDL on):  SCHED_PDL=1 clang++ ... -fpass-plugin=... \
//     test/pdl_overlap.cu -o pdl && ./pdl
#include "../runtime/sched_rt.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#define NSEQ 256
#define D 128
#define WORK 512  // producer inner iterations (gives PDL a tail to hide in)

__global__ void producer(float *buf, int work) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = i * 1e-6f;
  for (int k = 0; k < work; ++k)  // deterministic delay
    v = fmaf(v, 1.000001f, 1e-7f);
  buf[i] = v;
}

__global__ void consumer(const float *__restrict__ buf,
                         float *__restrict__ out) {
  int seq = blockIdx.x; // slot axis -> pi-remapped by the weave
  int d = threadIdx.x;
  out[seq * D + d] = buf[seq * D + d] * 2.0f;
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  return 1; } } while (0)

static int run_pair(float *buf, float *out, bool pdl, int iters, double *ms) {
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) {
    producer<<<NSEQ, D>>>(buf, WORK);
    if (pdl) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(NSEQ);
      cfg.blockDim = dim3(D);
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      CHECK(cudaLaunchKernelEx(&cfg, consumer, (const float *)buf, out));
    } else {
      consumer<<<NSEQ, D>>>(buf, out);
    }
  }
  CHECK(cudaDeviceSynchronize());
  *ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / iters;
  return 0;
}

int main() {
  int fails = 0;
  auto ok = [&](bool c, const char *m) {
    printf("  [%s] %s\n", c ? "PASS" : "FAIL", m);
    if (!c) ++fails;
  };
  printf("== PDL overlap fixture (griddepcontrol weave) ==\n");

  float *buf, *out_plain, *out_pdl;
  CHECK(cudaMalloc(&buf, sizeof(float) * NSEQ * D));
  CHECK(cudaMalloc(&out_plain, sizeof(float) * NSEQ * D));
  CHECK(cudaMalloc(&out_pdl, sizeof(float) * NSEQ * D));

  // Arm the host-ABI plane (identity pi) so the woven table reads are LIVE --
  // the latency PDL is supposed to hide.
  sched_rt_init();
  int32_t order[NSEQ];
  for (int i = 0; i < NSEQ; ++i)
    order[i] = i;
  sched_rt_set_order(order, NSEQ);
  sched_rt_set_num_tasks(NSEQ);
  sched_rt_push_ctrl();

  double ms_plain, ms_pdl;
  if (run_pair(buf, out_plain, false, 200, &ms_plain)) return 1;
  sched_rt_timer_clear();
  if (run_pair(buf, out_pdl, true, 200, &ms_pdl)) return 1;

  std::vector<float> a(NSEQ * D), b(NSEQ * D);
  CHECK(cudaMemcpy(a.data(), out_plain, sizeof(float) * a.size(),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(b.data(), out_pdl, sizeof(float) * b.size(),
                   cudaMemcpyDeviceToHost));
  ok(a == b, "PDL launch bit-exact vs plain launch (E0)");

  long long populated = 0;
  for (int i = 0; i < NSEQ; ++i)
    populated += sched_rt_timer(i) > 0;
  ok(populated == NSEQ, "woven timer populated under PDL");

  printf("  pair wall time: plain %.3f ms, PDL %.3f ms (%+.1f%%)\n",
         ms_plain, ms_pdl, 100.0 * (ms_pdl - ms_plain) / ms_plain);
  printf(fails == 0 ? "== ALL PASS ==\n" : "== FIXTURE FAILED ==\n");
  return fails;
}
