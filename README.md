# Validation of Stop Detection Algorithms on Simulated and Sparse GPS Mobility Data 

## Overview
We simulate a user trajectory and GPS pings produced at different frequencies. We generate a random synthetic trajectory of an individual doing visits of different durations to three locations of different sizes, two of which are in close proximity. To simulate the sparsity of the signals we subsample the complete trajectories to obtain sparse signals along with some noise representing horizontal inaccuracies in GPS data. The code written in Python, produces plots representing the outputs of two different parameterizations of a stop-detection algorithm. 

## Mobility Model
The synthetic trajectory simulates two hour-long visits to adjacent 30m radius locations followed by a trip to a 60m radius location for a three-hour visit. We simulate this by sampling from bivariate normals centered at each of the locations and the trips as bivariate normals with a mean that is moving along the path connecting the centers of the locations.

## Sparsity Model
We sample from a uniform distribution spanning the duration of the whole trajectory. Signals are sampled at "low" (3 Pings/Hour) and "high" (12 Pings/Hour) frequencies to demonstrate the impact of having heterogeneous sparsity patterns in the data.

## Stop Detection using DBScan
We apply two parameterizations to a stop-detection algorithm based on DBScan [1] which we adapted to resolve temporally-overlapping stops: “coarse” with parameters given by distance threshold d = 120 meters, time threshold t = 120 minutes, and minimum number of points N = 2; and “fine” with parameters given by distance threshold d = 75 meters, time threshold t = 60 minutes, and minimum number of points N = 3.

[1] Chen, Wen, M. H. Ji, and J. M. Wang. "T-DBSCAN: A Spatiotemporal Density Clustering for GPS Trajectory Segmentation." International Journal of Online Engineering 10.6 (2014).

## How to run
These simulations can be run by executing the jupyter notebook trajectory_simulation.ipynb. 

# Visualization of Safegraph's stay at home data for 2019-2020

## Overview
We visualize a 7-day rolling average of a stay-at-home metric (devices_completely_at_home/total_devices) using Safegraph's [social distancing metrics](https://docs.safegraph.com/docs/social-distancing-metrics) during 2019 and 2020. Access to the data requires authorization from Safegraph through their COVID-19 [data for Academics program](https://www.safegraph.com/blog/safegraph-partners-with-dewey). 
## How to run
These plots can be obtained by executing the jupyter notebook called safegraph_stay_at_home.ipynb