
# Online Context-Aware Task Assignment in Mobile Crowdsourcing via Adaptive Discretization  
  
This repository is the official implementation of Online Context-Aware Task Assignment in Mobile Crowdsourcing via Adaptive Discretization, published at [IEEE TNSE](https://ieeexplore.ieee.org/document/9894096). 

## Requirements
To install requirements:  
  
```setup  
pip install -r requirements.txt  
```  
  
We use the [gpflow](https://github.com/GPflow/GPflow) library for all GP-related computations and gpflow uses tensorflow. Our code uses the TIM+ algorithm, for which you must link the C++ TIM+ code to Python. Follow [here](https://github.com/altugkarakurt/OCIMP) for linking instructions. Once the library has been generated, place it both in the root directory where main.py is and also inside the tim_plus directory.  
  
## Running the simulations  
We ran a total of three simulations. Moreover, none of the algorithms that we implement and test do offline-learning, thus there is no 'training' to be done. However, to be able to repeat the simulations and also improve speed, we first generate the arm contexts, rewards, and other setup-related information and save them as HDF5, in the case of Simulation I and Simulation II, and pickled DataFrames, in the case of Simulations III. By default, when you run the script (main.py), it re-generates new datasets and runs the simulations on them.  
  
### Simulation I (Visualizing adaptive discretization)
To run Simulation I, provide the argument `simple_uni` to the main.py script for uniform context arrivals and `simple_nuni` for non-uniform arrivals.  
### Simulation II (DPMC)  
To run Simulation II, use the argument `dpmc`. You can provide the `--use_saved_sim` argument to use pre-generated and saved datasets.  
### Simulation III (Crowdsourcing)  
To run Simulation III, use the argument `gp`. You can provide the `--use_saved_sim` argument to use pre-generated and saved datasets.