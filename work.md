# Path-Planning and Automated Target Recognition

## Overview
Logos Technologies (Logos) builds autonomous drones that operate in a mission area independently to investigate targets of interest. Each drone uses automated target recognition and image projection to identify and localize areas of interest in real-time. Anduril, a defense company startup, is a competitor in this exact field, and I was responsible for developing the same algorithmic capabilities as Anduril. Specifically, I worked in two fields: (1) Path-Planning, and (2) Automated Target Recognition.

Path planning defines the navigation behavior of where and how the drone moves in a given mission area. In global path-planning, the drone needs specific waypoints to navigate to a location while avoiding keep-out zones. I researched and implemented an improvement to the A* search algorithm, visibility graphs. In surveillance, the drone needs specific waypoints to navigate towards such that each area in a mission area gets visited occasionally, while high-priority areas get visited more frequently. I researched and proposed a latency-graph representation of the problem, which provides an optimal route to minimize time between reacquiring areas of high priority. I also familiarized myself with motion planning, which uses convex optimization to find a desirable trajectory.

Automated target recognition is the algorithmic problem of recognizing targets of interest given image data. Convolutional neural networks are used ubiquitously due to the crucial translation-invariance property that the convolution function provides. I managed the Single-Shot-Detector (SSD), which uses a one-stage detection network to identify and localize targets.

The network had a good detection accuracy of 90% and ran at about 10 frames per second on an NVIDIA Xavier NX, but suffered from false alarms due to water glare. I implemented custom modifications to enable the network to run on low-power embedded hardware. Later, when the mission changed to allow for higher performance networks, I wrote data loaders and modified the network architecture of RTDETR, allowing for higher performance in situational contexts.

I provide details on all 3 path-planning algorithms that I have worked on in section 2, my work in automated target recognition in section 3, smaller projects that I worked on in section 4, and hobbyist projects that I worked on in my spare time in section 5.

## Path-Planning
The most essential function for an autonomous drone is path-planning. It must be able to move through space efficiently without colliding with any obstacles. There are generally 3 distinct problems within this framework: (1) Path Navigation, (2) Motion Planning, and (3) Surveillance.

Path navigation involves navigating from a source to a destination through obstacles. The goal is a sequence of waypoints that minimize the distance traveled to go from the source to the destination. Then a robot can go in a straight line to each waypoint, and when reached, travels in a straight line to the next waypoint until it reaches the destination.

Motion planning is similar to path navigation because the goal is still to navigate from a source position to a destination position, but differs from path navigation because it requires motor instructions for each timestep. Each motor requires unique instructions. The precise controls make the problem extremely difficult in space-constrained environments, where there is very little free space. Another difficult scenario is time-constrained environments, where the most critical success factor is time to target.

Surveillance is similar to path navigation in that the instructions involve a list of waypoints that the robot navigates towards in a straight line, but different in that there is no given starting point or ending point. The idea is that the drone should independently patrol a mission area, visiting important areas frequently while occasionally visiting other areas.

### Path Navigation
The foundational work in path navigation is the A* (A-star) search algorithm, which involves intelligently exploring locations to find a route from the source to the destination. The A* search algorithm always returns an optimal solution given an admissible heuristic function. A popular heuristic is distance, because it never overestimates the distance to the goal. However, the problem with the A* search is that when used as a path-planning algorithm, it doesn't quantize each grid cell intelligently, leading to extremely expensive computations for simple navigation areas.


The inefficiency comes from the fact that each grid cell is naively the same area even when it is known prior that there is no obstacle between them. This leads to A* spending lots of time in obvious segments. Originally, I was responsible for implementing this algorithm for drone navigation in 2D space. The idea behind the mission was to maintain a coherent world view with multiple patrolling drones while staying outside of known danger zones. With my prior knowledge in computational geometry and path-planning, I introduced my manager to visibility graphs and provided a mature implementation for the drone to use, introducing a significant algorithmic advantage over the A* search algorithm. It is hard to quantify how large the difference between the performance of A* and visibility graphs is. It allowed for a completely different representation. It turned 2D Euclidean space into a small graph with all the visibility relationships implicitly encoded. Complex areas that took minutes to find the best path now finished instantly.

Visibility graphs preprocess the Euclidean space given a list of obstacles. Each obstacle is represented by a list of points with the last point being identical to the first point. Visibility graphs convert the Euclidean space pathfinding problem to a graph-based pathfinding problem because of the axiom that the shortest path between two points is a straight line. If there is an obstacle in between the two points, then the shortest path will touch an obstacle. This axiom has a non-obvious conclusion: the shortest path can be found by traversing between the source, obstacle vertices, and the destination.

The naive algorithm is the simplest approach. It runs in O(n^4) time. Here, n represents the number of points: all obstacle vertices, the source, and the destination. It works by checking if traversing straight from any one vertex to another is a valid operation. The complexity comes from the fact that each vertex pair must be checked for visibility by iterating through every other vertex pair to check for any intersections. It takes n^2 iterations to check each vertex pair, and n^2 vertex pairs to check. It is considered valid if there are no obstacle vertices that cross between the points being checked.

The most difficult part happens when verifying whether any obstacle intersects the line. After the visibility graph is generated, it constructs a more efficient pathfinding representation. It is impossible for the middle of an obstacle edge to be part of the shortest path between two points.

Chan's GitHub repository implemented the basic idea of visibility graphs which was very useful to me. However, the code in the repository had a critical flaw. It automatically assumed that the shortest path would never involve staying on the polygon for two adjacent segments. There were no edges checked between points on the same polygon, only points going from one polygon to another. Any time there was only one polygon between the source and the destination, the algorithm would fail. I modified this work to add that adjacent points on the same polygon are visible.

### Surveillance
One of the ambiguous problems that I worked on at Logos was the surveillance problem. Given a large mission area, what is the best way to patrol the area so that the drone maintains situational awareness over the mission area? Some areas are more important than others and those should be visited more frequently. Less important areas don't need to be visited as much. The importance of each area is dynamic. At a particular point in time, one area can be extremely important only to become unimportant later on in the mission. Each area should get visited eventually, but the most important areas need to be visited more frequently.

After a few weeks of research, I discovered that latency graphs properly represent the problem at hand. Latency graphs are different from standard graphs because they have node weights and edge weights instead of just edge weights. The latency graph $ G = (\mathcal{V},\mathcal{E}) $ has edge lengths $ l $ and vertex weights $ \phi $. The vertex weights give the relative importance of each region for surveillance. The task is to find an infinite walk, which consists of a sequence of vertices $ (v_1,v_2,...) $, given that edge $ v_i,v_{i+1} \in \mathcal{E} $. The goal is to construct a walk $ W $ which repeatedly iterates through all the nodes in the graph for the minimum cost. The total cost of each walk is composed of the *weighted latency* of each vertex:

$$
C(W,v) := \phi(v)L(W,v)
$$

The objective is to minimize the cost of an infinite walk $ W $:

$$
C(W) := \max_{v\in\mathcal{V}}C(W,v)
$$

The graph has nodes representing important areas and edges representing the paths between them. Each node weight indicates the importance of the area, and each edge weight represents the travel time between areas. The goal is to minimize the maximum latency, which is the longest time any important area goes without being visited. By using latency graphs, I was able to develop an efficient patrol strategy that ensures critical areas are frequently monitored while still covering the entire mission area.

The algorithm works in a stride scheduling mechanism used in operating systems and combines this with TSP. Priorities are treated with exponential importance.

### Motion Planning
This section, unlike other sections, was still ongoing. I was abruptly laid off during the midpoint of this project. To extend a surveillance product into a weapon, we would need to use attritable fixed-wing drones to automatically converge on targets when a target is inside the convergence radius. Logos has a product that can launch autonomous drones and maintain a model of targets of interest. To keep up with companies like Anduril, surveillance drones weren't enough. Anduril already has this capability with their Fury drone. I was tasked with building up the foundational knowledge required for missile guidance.

This foundational research started off with proportional navigation while staying in the field of view, and then would transition into a UAV trajectory optimization problem similar to the work of Marcucci et al. They combine trajectory optimization with path planning and mixed-integer second-order cone optimization. I relaxed a bunch of assumptions made in that paper and slowly built up to the complexity of that problem.

The work started off with proportional navigation, where the drone has control over its change in angle and its acceleration. The drone would use the camera to track the target, and then use the proportional navigation equations to adjust its heading and acceleration to converge on the target. One of the challenges is that the drone would need to throttle its own speed to keep in the line of sight of the target. Another challenge was using the correct angle update that dealt with switching between 0 degrees and 360 degrees smoothly.

Then the idea was to connect this to trajectory optimization. Because there are no errors in navigation, given an initial state and a goal state, the optimal trajectory is deterministic. The trajectory optimization problem is then a matter of finding the optimal control inputs to the system. We can formulate this using the concept of Bezier curves. Bezier curves are polynomial representations of a trajectory. They consist of a sum of Bernstein polynomials. A Bernstein polynomial is defined as

$$ \beta_{k,d}(s) := \binom{d}{k}s^k(1-s)^{d-k} $$

One core idea in Bernstein polynomials is that they have a similar structure to the binomial distribution. The integral after summing up all the terms from k = 0 to d is 1. Bezier curves do exactly this. Formally, a bezier curve is defined as
$$
\gamma(s) := \sum_{k=0}^{d}\beta_{k,d}(s)\gamma_k
$$
A bezier curve is defined by the \( k \) hyperparameters called *control points*. Bezier curves are used because they generate smooth trajectories. The fewer control points in a bezier curve, the smoother the trajectory. Bezier curves are used to find closed solutions to problems that proportional navigation did.

Bezier curves can be utilized to enforce differential and motion constraints with nonlinear optimization. Common practice in research is to use nonlinear optimization software like MOSEK or Gurobi, but I found it much more intuitive to use PyTorch because it allows for precise control under the hood.


Automatic differentiation simplifies debugging the motion planner. The next steps would be to extend this in a simulation using the software in the loop autopilot simulation, to see if I'm able to implement a real version. I wouldn't actually need to finish developing the path-planning based version of this algorithm because it is intended to be used as missile guidance, not obstacle avoidance. This saves me a lot of work in getting a nonlinear, mixed-integer solver.

I'm not sure why the original paper insisted on a space decomposition for trajectory proposals. My intuition tells me that sampling-based planners like RRT or ant-colony optimization would have been much better. I wonder what I'm missing.

## Automated Target Recognition
Automated target recognition (ATR) refers to the class of problems that involve recognizing specified objects from imagery. One of the common use cases for object detection in military applications is for vehicles. Currently, the most popular way to approach the problem is with convolutional neural networks (CNNs).

Neural Networks approximate functions using a chain of nonlinear functions parametrized by coefficients. The coefficients can be updated to approximate any function if given an error signal. CNNs are used extensively for image recognition because of their translation invariance. The identity of the object does not change based on its location inside the image.

The performance of CNNs is measured by the accuracy and the speed. The accuracy is measured using the precision and recall metrics. The precision measures the probability of detection and the recall measures the probability of a true positive. The precision and recall do not account for the size of the objects in the image themselves, which is one weakness of the metric. The speed is measured using frames per second. How many frames of detections can be processed at once. CNNs are usually modified to achieve a balance between speed and accuracy by changing the depth of the network.

In practice, the largest factor that impacts the quality of CNNs is the quality of the data. Theoretically, there should be no difference between the training set and the evaluation set. Practically, this is not feasible and there is usually a large gap between the training distribution and the test distribution. However, there are a few approaches to mitigate the performance loss from this issue.

### Data Preparation
Any deviation of environment from the training set to the evaluation set will result in performance loss. In an attempt to mitigate this, game simulators are used. We used the Unreal Engine and Microsoft Flight Simulator to simulate the evaluation conditions.

#### Simulators
Unreal Engine is a game development platform that allows for hyperrealistic game simulations. The game simulations allow lighting control, camera parameter controls, and large landscapes tuning down to programmable specifications.

Our company built an internal Unreal Engine simulation to simulate the expected conditions for training. A drone would navigate 60-120 feet over nautical territory and capture imagery of boats. The drone would take navigation commands from a Python server. The navigation commands were in MAVLink format, a standardized drone communication format that interfaces with the server for state updates.

I wrote automation scripts to automate data collection with PowerShell. PowerShell is a Windows scripting language that allows task automation in Windows. I wrote a script that would set the simulation configuration files for a variety of different weather conditions, run the simulation for 10 minutes, and store the results on a server. This ended up cutting hours of time and countless attention. Then I ran a Python script to prepare it for the PyTorch data loader, manually verifying it before sending it to the data directory that we used for training.

Another type of simulation that we utilized was the Microsoft Flight Simulator. We would take imagery from Microsoft Flight Simulator and superimpose objects from the Unreal Engine onto it, providing accurate terrain information that the network can use to distinguish between signal and noise. I used bitmasking techniques to superimpose objects into the imagery.

#### Data Augmentation
After collecting internal data from previous runs in addition to the simulation, data augmentation is a popular technique to demonstrate semantic invariances to the network. The idea is to change the images in some way to show the network that certain transformations don't matter. A simple example of a data augmentation technique is flipping the image from left to right. Flipping the image does not change the identity of the object and helps the convolutional filters learn the objects from two perspectives.

Common practice (at least at Logos) was to use a chain of data augmentation methods including Resize, RandomCrop, RandomHorizontalFlip, and RandomPhotometricDistortion, which was extremely successful for object detection in evaluation scenarios.

### SSD
The network architecture of CNNs must play a balancing act between complexity and speed. Empirically, the best object detection backbones consist of 3 components: (1) backbone, (2) neck, and (3) head. The backbone is responsible for encoding the features into a set of feature maps at multiple resolutions. The neck modifies every resolution's feature maps to incorporate information from all the other resolutions. The head transforms the feature maps into bounding boxes and classifications.

The Single Shot Detector was extremely popular circa 2017 because it used a one-shot detection process. Prior networks first identified regions of interest and then classified them. The Single-Shot-Detector parallelizes the region proposal and classification network.

I used the mature implementation of the SSD provided in the *Advanced Deep Learning with Keras* textbook. Their implementation of SSD uses more successful layers like Batch Normalization. The VGG16 original paper (which SSD based the backbone off of) did not use any normalization techniques.


The network chained together AlexNet blocks, a popular feature map generation block at the time. The idea was to chain Conv-ReLU-DropOut-MaxPools together. However, our implementation used BatchNorm and removed the dropout. During the training process, I also used the data augmentation methods that I mentioned earlier.

The detection heads used the output from the last 3 layers. There are two components to each detection head: (1) classification and (2) regression. The classification heads used a modified version of cross-entropy loss called varifocal loss, which takes the class imbalance into account. The localization heads use a convolution to generate 4 parameters that localize a box $(c_x, c_y, \log{w}, \log{h})$. These don't actually encode the image-wide coordinates, which are stored separately in *anchors*. Instead, they are offsets with respect to the receptive field that each feature map represents.

For objects with different aspect ratios, the network would be forced to learn parameters with large magnitudes, an undesirable trait in neural networks. To combat this, instead of one box per region, there are multiple boxes per region, each with a different aspect ratio. That way, the network should easily learn at least one of the boxes and then the other boxes that correspond to the similar regions can be rejected during the non-max suppression step.

The final processing step, non-maximum suppression (NMS), consists of removing the objects below a certain confidence threshold and removing the boxes that overlap. The boxes with the highest confidence stay, and then there are hyperparameters for how much overlap is allowed before a detection is considered different.

The network was extremely efficient and performed adequately. The biggest issue with this network was the simplicity. Because the network was extremely simple with only two AlexNet Blocks for preprocessing, it could not capture more advanced patterns like glint. Glint is a whiteout effect that occurs when the lens is pointing towards the sun. The light would reflect off of the water and into the lens causing large portions of the image to be white.

To deploy a network in production, a model goes through 2 steps. The first step is to convert the .pth weights from PyTorch into an ONNX file. ONNX is an architecture-independent format that stores the weights and the model layer information. Usually, this is possible to do within torchvision. In TensorFlow, use the `tf2onnx` library to convert to ONNX. The next step is to convert the ONNX file into an architecture-specific format. I used TensorRT, which has network optimizations built in when using NVIDIA embedded systems for deployment. The command line tool `trtexec` is used to convert from the ONNX file to the PLAN file.

The most difficult part of deploying to embedded hardware is if the operations the network does are not supported. If this is the case, you need to build your own TensorRT Plugin. NMS was an extremely popular operation in networks before transformer-based networks, so all I had to do was call the BatchedNMS Plugin. After converting from the ONNX file to the plan file, incorporate the model into the event loop by placing the image into the device memory, running inference, and then collecting the output.

## Smaller Projects
### Image Projection
I briefly built Python simulations for the image projection problem. Image projection is the process of converting 3D points to 2D points given a camera model. The camera parameters include the focal length, the center of projection, and the distortion coefficients. There are typically two types of distortions: radial distortion and tangential distortion. Radial distortion is due to the lens being shaped like a meniscus lens, and tangential distortion is due to the lens not being perfectly aligned with the image plane.

### Image Registration
I ran DFT registration on internal datasets to verify its viability in bundle adjustment. Bundle adjustment is the process of aligning multiple images to a common coordinate frame given that the camera parameters are slightly different.
