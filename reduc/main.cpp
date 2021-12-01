#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>

constexpr size_t N_SAMPLES = 10'000'000;

[[nodiscard]] std::unique_ptr<std::vector<double>> init_vector(const uint64_t n)
{
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(0.0, 1.0);

  std::unique_ptr<std::vector<double>> res = std::make_unique<std::vector<double>>(n);

  for (auto &d: *res)
    d = distr(eng);

  return res;
}

namespace reduc
{
  [[nodiscard]] double reduc_c(const std::vector<double> &a)
  {
    double res = 0.0;

    const double *data = a.data();
    uint64_t len = a.size();

    for (uint64_t i = 0; i < len; ++i)
      res += data[i];

    return res;
  }

  [[nodiscard]] double reduc_for_range(const std::vector<double> &a)
  {
    double res = 0.0;

    for (auto &d: a)
      res += d;

    return res;
  }

  [[nodiscard]] double reduc_lambda(const std::vector<double> &a)
  {
    double res = 0.0;

    const auto lambda = [&res](const double &d) { res += d; };

    for_each(a.begin(), a.end(), lambda);

    return res;
  }

  [[nodiscard]] double reduc_reduce(const std::vector<double> &a)
  {
    return std::reduce(a.cbegin(), a.cend());
  }
};

template<typename FUNC_PTR>
void bench(const FUNC_PTR func, const std::vector<double> &a, const std::string str)
{
  const auto t1 = std::chrono::high_resolution_clock::now();
  double res_reduc = func(a);
  const auto t2 = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> ms = t2 - t1;

  std::cout << str << " res " << res_reduc << " took " << ms.count() << " ms\n";
}

int main(void)
{
  // Create vector
  std::unique_ptr<std::vector<double>>a = init_vector(N_SAMPLES);

  // Run benchmark
  bench(reduc::reduc_c, *a, "reduc_c");
  bench(reduc::reduc_for_range, *a, "reduc_for_range");
  bench(reduc::reduc_lambda, *a, "reduc_lambda");
  bench(reduc::reduc_reduce, *a, "reduc_reduce");

  // Release memory
  a.release();

  return 0;
}
