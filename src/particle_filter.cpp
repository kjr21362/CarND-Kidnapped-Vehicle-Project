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
using std::normal_distribution;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles
  std::default_random_engine gen;
  double std_x, std_y, std_theta;

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(0.0, std_x);
  normal_distribution<double> dist_y(0.0, std_y);
  normal_distribution<double> dist_theta(0.0, std_theta);
  for(int i=0; i<num_particles; i++) {
    double sample_x, sample_y, sample_theta;

    sample_x = x + dist_x(gen);
    sample_y = y + dist_y(gen);
    sample_theta = theta + dist_theta(gen);
    Particle p;
    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for(int i=0; i<num_particles; i++) {
    if(fabs(yaw_rate) < 0.0001){
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += (velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)));
      particles[i].y += (velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)));
      particles[i].theta += yaw_rate*delta_t;
    }
    
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    //std::cout << yaw_rate << "," << particles[i].x << "," << particles[i].y << "," << particles[i].theta << "," << particles[i].weight << std::endl;
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
  for(int i=0; i<observations.size(); i++){
    double ob_x = observations[i].x;
    double ob_y = observations[i].y;
    
    double min_dist = std::numeric_limits<double>::max();
    int min_id = -1;
    
    for(int j=0; j<predicted.size(); j++){
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      
      double new_dist = dist(ob_x, ob_y, pred_x, pred_y);
      if(new_dist < min_dist){
        min_dist = new_dist;
        min_id = predicted[j].id;
      }
  	}
    observations[i].id = min_id;
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
  for(int i=0; i<num_particles; i++) {
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta_p = particles[i].theta;
    
    vector<LandmarkObs> observations_m;
    for(int k=0; k<observations.size(); k++){
      double x_c = observations[k].x;
      double y_c = observations[k].y;
      double x_m = x_p + x_c * cos(theta_p) - y_c * sin(theta_p);
      double y_m = y_p + x_c * sin(theta_p) + y_c * cos(theta_p);

      LandmarkObs l_ob = {observations[k].id, x_m, y_m};
      observations_m.push_back(l_ob);
    }
    
    vector<LandmarkObs> predictions;
    for(int j=0; j<map_landmarks.landmark_list.size(); j++){
      float l_x = map_landmarks.landmark_list[j].x_f;
      float l_y = map_landmarks.landmark_list[j].y_f;
      double d = dist(x_p, y_p, l_x, l_y);
      if(d <= sensor_range){
        LandmarkObs l_ob = {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f};
        predictions.push_back(l_ob);
      }
  	}
    dataAssociation(predictions, observations_m);
    
    particles[i].weight = 1.0;
    for(int k=0; k<observations_m.size(); k++){
      double obs_x = observations_m[k].x;
      double obs_y = observations_m[k].y;
      int obs_id = observations_m[k].id;
      
      double mu_x, mu_y;
      for (int m=0; m<predictions.size(); m++){
        if(obs_id == predictions[m].id){
          mu_x = predictions[m].x;
          mu_y = predictions[m].y;
        }
      }
      particles[i].weight *= multiv_prob(std_landmark[0], std_landmark[1], obs_x, obs_y, mu_x, mu_y);
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  weights.clear();
  for(int i=0; i<num_particles; i++){
    weights.push_back(particles[i].weight);
  }
  std::default_random_engine gen;
  std::vector<Particle> new_particles;
  std::discrete_distribution<int> distribution (weights.begin(),weights.end());
  for(int i=0; i<num_particles; i++){
    int idx = distribution(gen);
    new_particles.push_back(particles[idx]);
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
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}