---
title: 'StableRLS: Stable Reinforcement Learning for Simulink'
tags:
  - Python
authors:
  - name: Robert Annuth
    orcid: 0000-0002-0862-56250
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Finn Nußbaum
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
  orcid: 0000-0002-0862-56250 # CORRECT THIS ONE 
    affiliation: 1
  - name: Christian Becker
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Technical University Hamburg, Germany
   index: 1
date: 08 May 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.

# Wird klar genug, dass es um RL geht
---

# Summary / Abstract
Innovative systems of the recent years highly depend on electronic circuits and their control. This complexity also comes with additional effort in development, testing and manufacturing. Therefore, it is often not economical to build the system and instead simulation based approaches are chosen. MATLAB [matlab] is commercial software that allows companys to perform quick prototyping, mathematical analyses and the Simulink extension allows to perform simulations. Also a reinforcement learning (RL) toolbox for MATLAB and Simulink is availabe. However, the toolbox is heavily under development and many recent RL innovations are not available since MATLAB is in general not the preffered choice for machine learning in academia and companies. Nevertheless, many companys and researchers use the great simulation capabilities of Simulink and build their custom simulation libraries. StableRLS allows to use those existing Simulink models for RL in Python. It handles the export of the model and provides a Python user interface build on the Gymnasium library for RL.

Clearly state the objectives of the paper and what readers can expect to learn from it.

# Introduction
Simulation-based approaches are actively used in various engineering applications, allowing for system modifications and instant observation of resulting behavioral changes. StableRLS (Stable Reinforcement Learning for Simulink) is a software framework that integrates Simulink simulation models into the Python library Gym (not Gymnasium) to enable reinforcement learning (RL). While the framework is presented from an electrical engineering perspective in the paper, Simulink is also utilized in communication engineering, control, signal processing, robotics, driving assistanrequirements. For example when working with external data for the environment, like weather data, it is necessary to read this between every simulation step. ce, and digital twins, making the framework relevant across multiple disciplines. StableRLS can be employed for disciplines beyond electrical grids, as long as Simulink models can be exported as C-code, which is typically feasible.

Simulink, a graphical editor, facilitates the modeling of various system components, providing a wide range of prebuilt blocks, algorithms, and physical systems. However, the RL interface in Simulink undergoes modifications with each release and sometimes exhibits unexpected behavior. In RL tasks, Python is commonly used in conjunction with Gym, a framework that offers a standardized API for communication between algorithms and environments (not Gymnasium). Environments typically involve simulations and can be challenging to create directly in Python. Several tools aim to simplify this error-prone process. Currently, there is no high-performing interface between Simulink and Python, emphasizing the need for StableRLS.

The paper is structured as follows: firstly, it provides a summary of the state-of-the-art, introduces existing frameworks, and presents a brief performance comparison. Subsequently, the main features of the StableRLS framework are outlined, and the API is introduced. For additional information, examples, and contributions, please refer to the documentation of the StableRLS package.

---

Simulation based approaches are actively used in various engineering applications and allow to modify systems and instantly observe the resulting changes in behavior. StableRLS (Stable Reinforcement Learning for Simulink) is a software framework integrating Simulink simulation models into the Python library Gymnasium [gymnasium] to perform RL. In paper the framework is presented from the perspective of electrical engineering but Simulink is for example also used in communication engineering, control, signal processing, robotics, driving assistance and digital twins. Therefore, this framework is not only relevant for the simulation of electrical grids but also other disciplines. StableRLS does not require any modifications to be used for any other discipline as long as the Simulink models can be exported as C-code, which is usually the case.
Simulink is a graphical editor that allows modelling all components of a system and it provides many prebuild blocks, algorihms and physical systems by default. However, the RL interface is modified at each release and sometimes does not behave as expected. Typically Python is used for RL tasks in combination with Gymnasium which is a framework providing a standard API to communicate between algorithms and environments [gymnasium]. The environment is typically a simulation and often difficult to create dicretcly in Python. Various different tools try to simplify the difficult process which is often prone to errors. Currently no performant interface between Simulink and Python is available pointing out the need for StableRLS. The paper is organized as follows. First the state of the art is summarized, existing frameworks are introduced and a short performance comparison is presented. Afterwards the main features of the framework are presented and the API is introduced. For further information, examples and contribution please have a look at the documentaiton of the StableRLS package [documentation].

RL is a field of machine learning that focuses on training one or multiple agents to make decisions in an environment to maximize a cumulative reward. The agent repetetly inteacts with the environment by means of actions and therewith influences the state of the environment. The agent receives feedback in form of rewards or penalties for each action. The goal of the agent is to learn a policy that lead to the highest possible cumulative reward. The research interests can divided in developing environments that the agent can interact with and developing algorithms for the agent to converge quickly to an optimal solution. Especially forumlating algorihms that are applicable to a vast set of environments is challenging. OpenAI developed a framework called gym [gym] that defines the API interface between the agent and the environment. Recently, OpenAI stopped the developement and the Farma foundation overtook the development by maintaining a package named Gymnasium [gymnasium] . One of the reasons for RL's growing popularity is its ability to tackle complex problems that are difficult to solve using traditional programming or supervised learning methods. RL excels in situations where the optimal solution is not known in advance, and the agent needs to learn through exploration and exploitation. For example it has found applications in robotics, healthcare, finance, energy management, logistics, game playing and many other domains. [source]





# State of the art
In the literature other authors already tried to solve the drawbacks of creating envionments for RL in Python. In [henry2021gym] the authors introduced a customizable Gym Open AI environment but it is focussed on electrical grids and the class based definition of the power grid is not reasonable for large grids. The authors in [marot2021learning], [fan2022powergym] and [henri2020pymgrid] proposed similar frameworks but all of them lack the expandability to other domains and usage of an easy to use modelling interface.
For example in [heid2020omg] the autors proposed a similar framework for the Modelica modelling environment. However, the learning curve of Modelica is steep creating the need for a easy to use modelling environment that can be combined with Gym to perform RL. Similar to this framework StableRLS uses Functional Mock-up Units (FMUs). An FMU is a standardized file format to exchange and integrate simulation models across diffent tools. An FMU contains a dynamical model combined with mathematical equations and has input and outputs. The implementation typically uses the Functional Mock-up Interface (FMI) standard, which defines data exhange formats and the structure.

# Proposed framework
StableRLS uses the internal capabilities of Matlab to compile a Simulink model to an FMU. Howerver, the capabilities of Matlab to compile FMUs are highly limites and require model modification. In general the effort to compile an existing model is so high that it makes more sense to start from scratch. StableRLS solves this issue and compiles existing Simulink models. Matlab also requires a user defined signal busstructure and especially for large simulation models the definition is error prone because all nested input signals have to be defined correctly and since the input to the simulation model correlates to the action structure of the agent it is often required to modify the signal structure while working on a RL problem. StableRLS creates the busstructure for any simulink model and also works with signals connected to multiple blocks, GoTo blocks and nested structures. As a result the user only has to create the environment model in Simulink and don't need to worry about the conversion of the model to an FMU.

While working on StableRLS we also put effort into comparing different methods to combine Simulink simulationmodels and Python Gymnasium. Nameley, the TCP/IP interface and the implemented Python interface. In gerneral Simulink doesn't allow to change simulation parameters during the simulation and since the simulation has to wait for the actions of the agent at each timestep we need to pause the Simulink simulation between each interaction of the agent. This is required for both interface options mentioned above. The difference is how the data between the Simulink Simulation and Python is exchanged. We found that pausing and restarting the simulation is mainly responsible for the performance and the time for datatrasmission can be neglected. Therefore, only the performance of StableRLS and the internal Python interface is compared in Table 1. It can be seen that StableRLS is XXXX times faster because it uses the benefits of FMUs. 
%TODO create table 1 with performance comparison

Figure 1 represents the general structure of the framework. The Simulink simulationmodel is converted to an FMU and integrated in StableRLS. Before the conversion can start the StableRLS package prepares the Simulink model since many requirements regarding the input and output signal definition must be met. The resulting FMU model is ready to run in any FMU simulator. In our case FMU interacts with StableRLS following the definitions of the gymnasium API to provide a environment for an agent. Nevertheless, the strucutre and behavior of the StableRLS package is as flexible as possible to account for different RL learning strategies and usecases. Please refere to our documentation [docu] for a detailed explanation of all features. 
To start the training process of the agent a config file must be specified containing the basic information about the environment (simulation duration, step time, path of the FMU) and this file is used to create a child of the StableRLS base class. In addition, all addtional parameters specified in this file are available within the resulting class. Finally, we implemented functionalities to account for specific requirements. For example when working with external data for the environment, like weather data, it is necessary to read this between every simulation step. Therefore, it is important to note that the simulation step size doesn't have to be identical with the step size of the agent. So, the simulation could run with a step time of 0.1 seconds but the agend only cooses an action every 1.5 seconds. To integrate external data into the model, the user can simply overwrite the existing function "????". Other examples for such general functionalities are the export of data and definition of action and observation spaces. Internally StableRLS is a custom environment for the Gymnasium Python package and exploits the functionalities of PyFMI [CITEEE] to simulate the FMU. 

![overview.svg]

# Example usecase

# Conclusion
In academia and also in industry AI methods are tested and checked if the suit specific use cases more and more often. To be able to check the usability of RL, for example, software packages like StableRLS are highly relevant because they provide an interface between existing simulation models and newly developed RL algorithms. The simple integration of Simulink models without additional effort to export the model or couple it with Python enables users to focus on improving the environment model and also to adapt or extend existing RL algorithms. In addition this framework not only has an impact on electrical engineering, which this framework was developed for originally. Without any adaptions the framework can also be used for other research disiplines as long as a Simulink model with inputs and outputs for the agent to interact with is available. Further development will be mainly in providing a more detailed use case gallery beyond the domain of electrical engineering to extend the visibility of the framework.




	•	Create FMU from Simulink models:
	•	The main feature of this framework is the ability to create FMUs from Simulink models with various input and output signals. The framework can handle complex signal and bus structures, including variant blocks and multiple blocks connected to one signal. It can also compile and export the FMU in a user-friendly manner.


