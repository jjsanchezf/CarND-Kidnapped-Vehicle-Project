/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<double> pos_x(x, std[0]);
  std::normal_distribution<double> pos_y(y, std[1]);
  std::normal_distribution<double> pos_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = pos_x(gen);
    p.y = pos_y(gen);
    p.theta = pos_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(1.0);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
  double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);

  double yaw_dt = yaw_rate * delta_t;
  double velyaw = velocity / yaw_rate;
  double vel_dt = velocity * delta_t;

  for (auto& p : particles) {
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);

    if (fabs(yaw_rate) <= 0.0001) {
      p.x += vel_dt * cos_theta + noise_x(gen);
      p.y += vel_dt * sin_theta + noise_y(gen);
    }
    else {
      p.x += (velyaw * (sin(p.theta + yaw_dt) - sin_theta)) + noise_x(gen);
      p.y += (velyaw * (cos_theta - cos(p.theta + yaw_dt))) + noise_y(gen);
    }
    p.theta += yaw_dt + noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
  vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (auto& obs : observations) {
    obs.id = -1;
    double min_dist = std::numeric_limits<float>::infinity();

    for (auto pred : predicted) {
      double distance = dist(obs.x, obs.y, pred.x, pred.y);

      if (distance < min_dist) {
        min_dist = distance;
        obs.id = pred.id;//j;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  const vector<LandmarkObs> &observations,
  const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double std_x_sq_2 = 2.0 * std_x * std_x;
  double std_y_sq_2 = 2.0 * std_y * std_y;
  double norm = 1.0 / (2.0 * M_PI * std_x * std_y);
  weights.clear();
  for (auto& p : particles) {

    // step 1: collect valid landmarks
    vector<LandmarkObs> landmark_candidates;
    for (auto lm : map_landmarks.landmark_list) {
      double distance = dist(p.x, p.y, lm.x_f, lm.y_f);

      if (distance <= sensor_range) {
        landmark_candidates.push_back(LandmarkObs{ lm.id_i, lm.x_f, lm.y_f });
      }
    }

    // step 2: convert observations coordinates from vehicle to map
    vector<LandmarkObs> map_observations;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);

    for (auto obs : observations) {
      LandmarkObs tmp;

      tmp.x = p.x + cos_theta * obs.x + (-sin_theta * obs.y);
      tmp.y = p.y + sin_theta * obs.x + cos_theta * obs.y;
      tmp.id = obs.id;
      map_observations.push_back(tmp);
    }

    // step 3: find landmark index for each observation
    dataAssociation(landmark_candidates, map_observations);

    // step 4: compute the particle's weight:
    p.weight = 1.0;
    for(const auto& obs_m: map_observations){  
      if (obs_m.id > 0) {
        Map::single_landmark_s nearest = map_landmarks.landmark_list.at(obs_m.id - 1);
        double x_diff = obs_m.x - nearest.x_f;
        double y_diff = obs_m.y - nearest.y_f;
        double x_diff_sq = x_diff * x_diff;
        double y_diff_sq = y_diff * y_diff;
        double exponent = (x_diff_sq / std_x_sq_2) +
          (y_diff_sq / std_y_sq_2);
        exponent = exp(-exponent);
        double weight_add = norm * exponent;
        p.weight *= weight_add;
      }
    }
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  discrete_distribution<> dist(weights.begin(), weights.end());
  vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[dist(gen)]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
  const vector<int>& associations,
  const vector<double>& sense_x,
  const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  }
  else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}