# Cluster Scaling Policy Based on Logistic Regression

## Introduction

This project aims to implement a machine learning-based cluster scaling policy using Logistic Regression. The goal is to predict the number of nodes required in the cluster at any given time of the day, maximizing the utilization of CPU, GPU, and Network resources, and minimizing the task rejection rate.

## Features

The Logistic Regression model uses the following features from each task:

- Load (int): From 1 to 10, indicates the intensity of the load.
- CPU (float): CPU utilization.
- Memory (float): Memory utilization.
- Network (float): Network utilization.
- Arrival Time (int): The time when the task arrives.
- Deadline (time_window): The time by which the task needs to be completed.

Additionally, we engineer new features that better represent the relationship between the task object and the number of nodes, such as the ratio of `cpu` to `load`, or the difference between `deadline` and `arrival_time`.

The target variable is the number of nodes in the cluster at any given time.

## Data

Historical data is collected .

## Model

We train a Logistic Regression or Decision Tree model on this data to predict the number of nodes required in the cluster based on the task features and time of the day. The model is validated using a separate test set to ensure that it generalizes well to new data.

## Questions
1. **Trainng data** 
   1. Input: task object, time of the day.
   2. Output: number of nodes.
2. **Task Scheduler**
   1. If node scales up or down, task needs to be redistributed. The scaling policy needs to be mild to reduce the shuffle. Load will be an integer number.
   2. Don't reschedule the running tasks, only reschedule the future tasks
3. **Load Generation**
   1. Load generation scripts are written in java, how to pass information to the processing script written in python
      1. Python reads a textfile at the speed of poisson distribution