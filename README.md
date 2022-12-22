# PINNS_for_undulating_bodies

Thank you for visiting this repository, which contains the codes necessary to utilize a physics-informed neural network (PINN) to predict the 2D instantaneous pressure field around an undulating body from 2D Particle Image Velocimetry (PIV) measurements. The general idea of the proposed method is to use the PINN to predict an optimal velocity and pressure field that satisfies, in an L2 sense, the Navier-Stokes equations and the constaints put forth by the measurements. Please note that since the PINN is primarily being used as an optimizer, it is only applicable to the specific problem one is looking to analyze. It is not meant to be a robust network that can predict velocity and pressure fields for a wide range of applications. 

The primary codes are PINN_2D_UndulatingBodies.py and utilities.py. Also included is a MATLAB that will take the input text files and convert them into the necessary MATFILEs required to run the PINN. Please refer to the tutorial to determine the format of the initial input files to the GUI, how to use the GUI, and how to set up the necessary environement and to run the PINN. A sample data set (2D Simulation Data) is provided to use as practice. 

If you have any questions please feel free to email me at mcalicc1@jhu.edu
