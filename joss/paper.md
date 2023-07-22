---
title: 'StableRLS: Stable Reinforcement Learning for Simulink'
tags:
  - Python
authors:
  - name: Robert Annuth
    equal-contrib: false
    orcid: 0000-0002-0862-5625
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Finn Nussbaum
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    orcid: 0000-0003-2512-8134
    affiliation: 1
  - name: Christian Becker
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Institute of Electrical Power and Energy Technology, Hamburg University of Technology, Germany
   index: 1
date: 22 July 2023
bibliography: paper.bib
---

# Abstract
Today's innovative systems rely heavily on complex electronic circuitry and control. This complexity is associated with additional development, test and manufacturing efforts. As a result, it is often not economical to build the first prototype of a system in hardware and simulation-based approaches are chosen instead. MATLAB [@matlab] is commercial software that enables companies to perform rapid prototyping, mathematical analysis and provides the possibility to perform simulations with the Simulink toolbox. A reinforcement learning (RL) toolbox is also available for MATLAB and Simulink. However, the toolbox is still under development and many of the last RL innovations are not available because MATLAB is often not the preferred choice for machine learning in academia and industry. Nevertheless, many companies and researchers use the powerful simulation capabilities of Simulink to build their own simulation libraries. StableRLS allows them to use these existing Simulink models to perform RL in Python. It handles the conversion of the model and provides a Python user interface based on the Gymnasium library for RL.

# Introduction
Simulation-based approaches are actively used in various engineering applications. They make it possible to modify systems and immediately observe the resulting changes in behavior. StableRLS (Stable Reinforcement Learning for Simulink) is a software framework that integrates Simulink simulation models into the Python library Gymnasium [@gymnasium] to perform reinforcement learning (RL). In this paper the framework is presented from the perspective of electrical engineering, but Simulink is also used in many other disciplines, including communication engineering, control, signal processing, robotics, driver assistance and digital twins. Therefore, this framework is relevant not only to electrical grids simulation, but also to other disciplines. StableRLS does not require any modifications to be used for other disciplines, as long as the Simulink models can be exported as C code, which is usually the case.

Simulink is a graphical editor for modeling various components of a system, and it provides many pre-built blocks, algorithms and physical systems by default. However, the RL interface is modified with each release and sometimes does not behave as expected. Instead, Python is often used for RL tasks in combination with Gymnasium [@gymnasium], which is a framework providing a standard API for communication between RL algorithms and environments. The environment is usually a simulation and often difficult to create directly in Python. Various tools try to simplify this difficult and error-prone process. Currently, there is no high-performance interface between Simulink and Python available, highlighting the need for StableRLS. The paper is organized as follows. 

First, we summarize the state of the art, introduce existing frameworks and present a brief performance comparison. Then, the main features of the framework are presented and the API is introduced. Finally, an example from the field of electrical engineering presented. Additional information, examples, and a guide for contributors, can be found in the documentation of the StableRLS package.


## Reinforcement Learning Overview
RL is a field of machine learning that focuses on training one or more agents to make decisions in an environment to maximize their cumulative reward. The agent repeatedly interacts with the environment by performing actions, thereby influencing the state of the environment. The agent receives feedback in the form of rewards or penalties for each action. The goal of the agent is to learn a policy that leads to the highest possible cumulative reward. The research interests can be divided into developing environments that the agent can interact with, and developing algorithms for the agent to quickly converge to an optimal solution by means of the policy. In particular, formulating algorithms that are applicable to a vast set of environments is challenging. OpenAI developed a framework in 2016 called gym [@gym] that defines the API interface between the agent and the environment. Recently, OpenAI stopped the development and the Farama Foundation took over the development by maintaining a package called Gymnasium [@gymnasium]. One of the reasons for RL's growing popularity is its ability to tackle complex problems that are difficult to solve using traditional programming or supervised learning methods. RL excels in situations where the optimal solution is not known in advance, and the agent needs to learn through exploration and exploitation. For example, it has found applications in robotics, healthcare, finance, energy management, logistics, game playing and many other domains. [@sutton2018reinforcement]


# State of the Art
In the literature, other authors have already tried to simplify the process of creating environments for RL in Python. In [@henry2021gym] the authors introduced a customizable Gym OpenAI environment, but it's focused on electrical networks and the class-based definition of the power network is not reasonable for large network since each component and structure has to be defined separately. The authors in [@marot2021learning], [@fan2022powergym] and [@henri2020pymgrid] have proposed similar frameworks, but all of them lack extensibility to domains other than electrical engineering and don't include an easy-to-use modeling interface.
For example, in [@heid2020omg] the authors proposed a framework called OMG which is similar to StableRLS for the Modelica modeling environment [@fritzson2018openmodelica]. However, due to the steep learning curve of Modelica, there is a need for a framework that relies on another modeling environment that can be combined with Gymnasium to perform RL. Similar to the OMG framework, StableRLS uses Functional Mock-up Units (FMUs). A FMU is a standardized file format to exchange and integrate simulation models between different simulation tools. An FMU contains a dynamic model combined with mathematical equations and has inputs and outputs. The StableRLS implementation uses the Functional Mock-up Interface (FMI) 2.0 standard [@FMI2020], which defines the data exchange formats and structure.

# Proposed Framework - StableRLS
StableRLS uses the internal capabilities of MATLAB to compile a Simulink model to a FMU. However, the capabilities of MATLAB to compile FMUs are highly limited and require model modification. In general, the effort to compile an existing model is so high that it often makes more sense to start from scratch. StableRLS solves this issue by modifying Simulink models automatically and compile those models to a FMU. MATLAB requires a user defined signal bus structure and especially for large simulation models the definition is error-prone because all nested input signals have to be defined correctly and since the input to the simulation model correlates to the action structure of the agent it's often required to modify the signal structure while working on a RL problem. StableRLS creates the bus structure for any Simulink model and also works with signals connected to multiple blocks, GoTo blocks and nested structures. As a result, the user only has to create the environment model in Simulink and don't need to worry about the conversion of the model to a FMU.

While working on StableRLS we also put a lot of effort into comparing different methods to combine Simulink simulation models and Python Gymnasium. Namely, the TCP/IP interface and the implemented Python interface which is called MATLAB engine. In general, Simulink doesn't allow changing simulation parameters during the simulation, so the simulation has to wait for the actions of the agent at each time step. As a result, it's required to pause the Simulink simulation between each interaction of the agent. This is the case for both interface options mentioned above. The only difference is how the data between the Simulink simulation and Python is exchanged. We found that pausing and restarting (start-stop-method) the simulation mainly determines the resulting performance, and the time for data transmission can be neglected. Therefore, the performance of the MATLAB engine and the TCP/IP method is almost identical and only the performance of StableRLS package and the internal Python interface is compared in the table below. It can be seen that StableRLS is above 900 times faster because it uses the benefits of FMUs. 

: Performance comparison between this package and another common method to use reinforcement learning with Simulink models. The simulated time was 500 seconds, with a step size of 0.2 seconds.

+-------------------+-----------------+
|                   | Computation Time| 
+===================+=================+
| StableRLS         | 0.049s          |
+-------------------+-----------------+
| Start-stop-method | 45.13s          |
+-------------------+-----------------+

\autoref{fig:overview} represents the general structure of the framework. The Simulink simulation model is converted to a FMU and integrated into StableRLS. Before the conversion can begin, the StableRLS package prepares the Simulink model since many requirements regarding the input and output signal definition must be met. The resulting FMU model is ready to run in any FMU simulator. In our case, the FMU interacts with StableRLS following the definitions of the Gymnasium API [@gymnasium] to provide an environment for an RL agent. Nevertheless, the structure and behavior of the StableRLS package is as flexible as possible to account for different RL learning strategies and use cases. Please refer to our documentation for a detailed explanation of all features. 

To start the training process of the agent a configuration file must be specified containing the basic information about the environment, e.g. simulation duration, step size, path of the FMU. This file is used to create a child of the StableRLS base class. In addition, any other parameters specified in this file are available within the resulting class. Finally, we implemented functionalities to account for specific requirements. For example when working with external data for the environment, like weather data, it's necessary to read this between every simulation step. Therefore, it's important to note that the simulation step size doesn't have to be identical with the step size of the agent. As a result, the simulation could run with a step size of 0.1 seconds but the agent only chooses an action every 1.5 seconds. To integrate external data into the model, the user can simply overwrite the existing function "FMU_external_input(self)". Other examples of such general functionalities are the export of data and the definition of action and observation spaces. Internally StableRLS is a custom environment for the Gymnasium Python package and exploits the functionalities of PyFMI [@andersson2016pyfmi] to simulate the FMU. 

![General overview of the software package and how it interacts with MATLAB Simulink and the Python Gymnasium environment.\label{fig:overview}](overview.pdf){ width=1400px }

# Example Use Case: Electrical Microgrid
To demonstrate the usage of the package, we chose a small electrical grid with a nominal voltage of 48V as an environment which is controlled by an RL agent. In this paper, we only provide a rough overview about the required steps to work with the StableRLS package and run the simulation. The example is also included in the software repository with additional information. 
It is demonstrated how the RL agent coordinates different electrical energy sources to feed a load with constant power by using the droop concept. In this case we use a linear droop which results in a decrease in the output voltage of the voltage source and the battery. 

The RL agent can modify the voltage reference of the PV-array and the battery relative to nominal value. The action space of the agent is two-dimensional and has 11 discrete values to choose from. E.g. an action with the value 5 refers to not changing the voltage reference and any increase or decrease of the action value will modify this reference. Not all observations of the environment are visible to the agent, and the example also shows how to scale and process them. At each time step the agent interacts with the simulation it observes the voltages and currents of all four components and in addition the state-of-charge of the battery. 

\autoref{fig:model} shows the structure of the Simulink simulation model. A PV-array is used to generate renewable energy which can be consumed directly by the constant power load or can be stored in the battery. The amount of load which is not covered by these two sources is fed by a backup voltage source. However, the energy from this voltage source should be as small as possible. 


![Simulation model of the example use case, containing one load and three sources connected by electrical lines.\label{fig:model}](model.pdf){ width=1400px }

The example in the software repository contains further details how to configure the agent, because the FMU simulation has a step size of 1 millisecond and the RL agent can interact with the environment every 10 seconds. Also, irradiance data is used for the PV-array to calculate the energy production. This data has a sampling time of 1 minute and has to be updated in the simulation.
Additionally, to the actions of the agent, the load, irradiance and temperature of the PV-array are set internally every simulation step. However, as mentioned above, these are not observed by the agent. \autoref{fig:res_input} shows the load data and also the irradiance data of the PV-array.

![Power of the constant power load and irradiance of the PV-array used as input for the simulation.\label{fig:res_input}](result_input.pdf){ width=1400px }

For simplicity reasons and because the training itself goes beyond the scope of the StableRLS package and this paper, we decided to run the agent with fixed actions instead of choosing an algorithm to find the optimal actions. However, this is easily possible by training e.g. an PPO agent [@schulman2017proximal] with each action, observation and reward. We run the simulation for one episode, which is 15 minutes long. [@gymnasium]

For this demonstration, we chose for both action values 5. As a result, the reference voltage is always equal to the voltage nominal voltage. So, we would expect almost identical output voltages of the battery and PV-array. The output voltage is not identical, since the battery voltage deviates slightly from the nominal voltage of 48V in dependence of the state of charge.

![Simulated voltages and currents of the four components within the simulated electrical grid.\label{fig:res_ui}](result_ui.pdf){ width=1400px }

\autoref{fig:res_ui} shows the results of the simulation. Because the agent doesn't change the voltage reference by choosing a constant action, the voltage of the battery and the voltage source are also almost identical as expected. The power of the PV-array oscillates around the power of the load. When the power from the PV-array is lower compared to the load, additional power from the grid and the battery is used. The power is shared almost identical since, the droop reference voltage is not modified in this example and the droop curve is identical. If the PV-power exceeds the load demand, the additional power is feed back into the grid and the battery. While charging and discharging of the battery, the voltage deviates slightly from the normal voltage leading to a small difference between the current and voltage of the voltage source and the battery. Last, the droop behavior is also visible within the voltage plot. At high loads, the output voltage of all components decreases with respect to the droop coefficient.

# Conclusion
In academia and industry, AI methods are increasingly being tested and validated for their suitability for specific use cases. To verify the applicability of RL, software packages such as StableRLS are highly relevant because they provide an interface between existing simulation models and newly developed RL algorithms. The easy integration of Simulink models without additional effort to export the model or couple it to Python, allows users to focus on improving the environment model and also to adapt or extend existing RL algorithms. In addition, this framework has implications beyond electrical engineering, for which it was originally developed. The framework can also be used for other research disciplines without any adaptations, as long as a Simulink model is available for the agent to interact with. Further development will focus on providing a more detailed gallery of use cases beyond the domain of electrical engineering to increase the visibility of the framework.
