#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
using namespace mlpack;
using namespace mlpack::distribution;
using namespace mlpack::gmm;

int main()
{
  /*
  GMM gmm(2, 2);
  gmm.Weights() = arma::vec("0.40 0.60");

  // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
  gmm.Component(0) = distribution::GaussianDistribution("1.5 5.0",
      "1.00 0.60; 0.60 0.89");


  // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
  gmm.Component(1) = distribution::GaussianDistribution("4.50 0.5",
      "1.00 0.70; 0.70 1.01");
  
  // Now generate a bunch of observations.
  arma::mat observations(2, 1000);
  for (size_t i = 0; i < 1000; i++)
    observations.col(i) = gmm.Random();
  */

  GMM gmm(5, 2);
  gmm.Weights() = arma::vec("0.10 0.20 0.20 0.30 0.20");

  // N([2.25 3.10], [1.00 0.20; 0.20 0.89])
  gmm.Component(0) = distribution::GaussianDistribution("1.5 5.0",
      "1.00 0.60; 0.60 0.89");


  // N([4.10 1.01], [1.00 0.00; 0.00 1.01])
  gmm.Component(1) = distribution::GaussianDistribution("4.50 0.5",
      "1.00 0.70; 0.70 1.01");
  

  gmm.Component(2) = distribution::GaussianDistribution("0.00 0.00",
      "1.00 0.60; 0.60 0.89");
  

  gmm.Component(3) = distribution::GaussianDistribution("6.0 4.5",
      "1.00 0.70; 0.70 1.01");
  

  gmm.Component(4) = distribution::GaussianDistribution("-2.0 2.0",
      "0.50 0.20; 0.20 0.50");
  
  // Now generate a bunch of observations.
  arma::mat observations(2, 2000);
  for (size_t i = 0; i < 2000; i++)
    observations.col(i) = gmm.Random();

  data::Save("multiple_5.csv", observations);
}