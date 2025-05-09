#include <torch/extension.h>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <iostream>

//compile option: g++ lesson8.cpp -march=native  -o lesson8  //-march=native will turn on all the feature flags that your own machine supports
//GCC编译器和ICC编译器编译intrinsic的区别：如果使用ICC，那么不需要指定-march=native
//https://stackoverflow.com/questions/44962849/inlining-failed-in-call-to-always-inline-m256d-mm256-broadcast-sdconst-doub
//https://stackoverflow.com/questions/55747789/the-effect-of-architecture-when-using-sse-avx-intrinisics
//运行前准备工作：lscpu看看当前cpu有没有avx512f flag
//fp32 rmsnorm
// src: {rows, cols}
void RMSnorm_fp32(float* src, float* gamma, float* dst, float ln_eps, int rows, int cols)
{
  int M = rows;
  int N = cols;
  auto len = _mm512_set1_ps((float)(N)); //把1个fp32 broadcast到1个avx512 reg
  auto eps = _mm512_set1_ps(ln_eps);
 
  for (int i = 0; i < M; ++i)
  {
    // 初始化一个为0的avx512 reg
    for (int j = 0; j < N; j += 16)
    {
      //把fp16转换为fp32
      //向量除法，dat / len
       //fma, 即ele1 * ele1 / len
    } // 各个向量求各自的x^2 / len, 最后累加, inter avx512 vec reg
 
 
    for (int j = 0; j < N; j += 16)
    {
      //分子除分母
       //结果由res reg存储到对应offset
    }
  }
}
// src: {rows, cols}, 映射到图片的{N,C,H,W}格式的话rows = N, cols = C*H*W
// y=(x-E(x))/sqrt(var(x)+eps) * gamma + beta
// var(x) = E(x^2)-E(x)^2
// 0.1708
void layernorm_avx(int rows, int cols, float *src, float *gamma,
                   float *beta, float *dst, float &ln_eps)
{
  auto len = (float)(cols);
  // 为了满足avx512 intrinsic的输入类型要求，以下三行把0，1，eps全部广播为_m512类型
  auto zero = _mm512_setzero_ps();
  auto one = _mm512_set1_ps(1.f);
  auto eps = _mm512_set1_ps(ln_eps);
 
  for (int i = 0; i < rows; ++i)
  {
    // Calculate Mean for each row
    auto sum_mean = _mm512_setzero_ps();
    auto sum_var = _mm512_setzero_ps();
    for (int j = 0; j < cols; j += 16)
    {// Ex和E(X^2)
      // dat = __m512
      auto dat = _mm512_loadu_ps(src + i * cols + j);
      sum_mean = _mm512_add_ps(sum_mean, dat);
      // a*b+c,fma=fused multiply and add
      sum_var = _mm512_fmadd_ps(dat, dat, sum_var);
    } // N / 16 个mean和var
    // 16个元素reduce并除以E(x)
    float mean_val = _mm512_reduce_add_ps(sum_mean) / len;
    // E(x^2) - E(x)^2
    float var_val = _mm512_reduce_add_ps(sum_var) / len - mean_val * mean_val;
    // broadcast mean_val to 16x mean_val
    auto mean = _mm512_set1_ps(mean_val); 
    // 1 / sqrt(var + eps)
    auto var = _mm512_div_ps( 
        one, _mm512_sqrt_ps(_mm512_add_ps(
                 eps, _mm512_max_ps(zero, _mm512_set1_ps(var_val)))));
 
    // LayerNorm
    for (int j = 0; j < cols; j += 16)
    {
      // (1 / sqrt(var + eps)) * gamma
      auto amplifier =
          _mm512_mul_ps(_mm512_loadu_ps(gamma + j), var);
      // x
      auto dat = _mm512_loadu_ps(src + i * cols + j);
       // x - Ex
      auto x_mean = _mm512_sub_ps(dat, mean);
      // a * b + c = fma
      auto dst_val = _mm512_fmadd_ps(amplifier, x_mean, _mm512_loadu_ps(beta + j));// (x - E(x) / sqrt(var + eps)) * gamma - beta
      // consider inplace to save allocate dst buffer
      _mm512_storeu_ps(dst + i * cols + j, dst_val);
    }
  }
}
// CPU AVX512 layernorm kernel
torch::Tensor layernorm_cpu(torch::Tensor src, torch::Tensor gamma, torch::Tensor beta) {
    auto rows = src.size(0);
    auto cols = src.size(1);
    float ln_eps = 1e-5;
    torch::Tensor dst = torch::empty({rows, cols});
    layernorm_avx(rows, cols, src.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), dst.data_ptr<float>(), ln_eps);
    return dst;
}
// void LayNorm_main() {
//   int rows = 100;
//   int cols = 2048;
//   float* src = (float*)malloc(rows * cols * sizeof(float));
//   float* gamma = (float*)malloc(cols * sizeof(float));
//   float* beta = (float*)malloc(cols * sizeof(float));
//   float* dst = (float*)malloc(rows * cols * sizeof(float));
//   float ln_eps = 1e-5;
//   // initialize
//   for(int i = 0; i < rows * cols; i++) {
//     src[i] = (float)(i % 4 + 1); // 1 2 3 4 1 2 3 4...
//     if(i < cols) {
//       gamma[i] = (float)((i % 4 + 1) * 0.5);
//       beta[i] = (float)((i % 4 + 1) * 0.5);
//     }
//   }
//   //  call kernel
//   layernorm_avx(rows, cols, src, gamma, beta, dst, ln_eps);
//   std::cout << "layernorm output: " << dst[0] << std::endl;
//   free(src);
//   free(gamma);
//   free(beta);
//   free(dst);
// }
// 定义Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &layernorm_cpu, "layer normalization");
}