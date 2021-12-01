#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>

constexpr size_t N_SAMPLES = 10'000'000;

[[nodiscard]] std::vector<double> init_vector(const uint64_t n)
{
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(0.0, 1.0);

  std::vector<double> res(n);

  for (auto &d: res)
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

  [[nodiscard]] double reduc_openmp_c(const std::vector<double> &a)
  {
    double res = 0.0;

    const double *data = a.data();
    uint64_t len = a.size();

#pragma omp parallel for reduction(+:res)
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

  [[nodiscard]] double reduc_openmp_for_range(const std::vector<double> &a)
  {
    double res = 0.0;

#pragma omp parallel for reduction(+:res)
    for (auto &d: a)
      res += d;

    return res;
  }

  [[nodiscard]] double reduc_iterator(const std::vector<double> &a)
  {
    double res = 0.0;

    for (auto it = a.begin(); it < a.end(); ++it)
      {
        res += *it;
      }

    return res;
  }

  [[nodiscard]] double reduc_openmp_iterator(const std::vector<double> &a)
  {
    double res = 0.0;

#pragma omp parallel for reduction(+:res)
    for (auto it = a.begin(); it < a.end(); ++it)
      {
        res += *it;
      }

    return res;
  }

  [[nodiscard]] double reduc_for_each(const std::vector<double> &a)
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
  std::vector<double> res_reduc(50);

  const auto t1 = std::chrono::high_resolution_clock::now();
  for (auto &res: res_reduc)
    res = func(a);
  const auto t2 = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> ms = t2 - t1;

  if (std::all_of(res_reduc.begin(), res_reduc.end(),
                  [&res_reduc](auto &res)
                  {
                    return (res == res_reduc[0] ? 0 : 1);
                  }))
    {
      exit(1);
    }

  std::cout << std::setw(25) << std::left << str << " result: " << res_reduc[0] << ", took: " << ms.count() << " ms\n";
}

int main(void)
{
  // Create vector
  std::vector<double> a = init_vector(N_SAMPLES);

  // Run sequential benchmark
  bench(reduc::reduc_c, a, "reduc_c");
  bench(reduc::reduc_for_range, a, "reduc_for_range");
  bench(reduc::reduc_iterator, a, "reduc_iterator");
  bench(reduc::reduc_for_each, a, "reduc_for_each");

  //
  std::cout << std::endl;

  // Run openmp benchmark
  bench(reduc::reduc_openmp_c, a, "reduc_openmp_c");
  bench(reduc::reduc_openmp_for_range, a, "reduc_openmp_for_range");
  bench(reduc::reduc_openmp_iterator, a, "reduc_openmp_iterator");

  //
  std::cout << std::endl;

  // Run reduce benchmark
  bench(reduc::reduc_reduce, a, "reduc_reduce");

  return 0;
}
