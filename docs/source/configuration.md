# Python Configuration
The config is supposed to simplify the user interaction with the package by providing a central place to set all parameters. By default the config file has to contain three sections:

```
[General]
[FMU]
[Reinforcement Learning]
```

The parameters within the section will be converted to the respective type and are available within the environment class.

## Required Parameters
The values are examples to show the required type.

| Name | Value | Description|
|---|---|---|
| FMU_path | path/to/fmu| relative or absolute path to the file|
| stop_time | 1 | when the time is reached the simulation ends [s]|
| dt | 0.5 | fixed timestep of the simulation [s]| 

## Optional Parameters

| Name | Value | Description|
|---|---|---|
|action_interval|1.5|every 1.5 seconds the agent can take an action [s]|
|reset_inputs| True | if this is True the inputs are reset to 0 between each episode|
|start_time | 1 | specify the start time of the FMU simulation [s]|

Additional parameters will be also available as `self.parameter`.

## Example Config
```
[General]
[FMU]
FMU_path = 00-Simulink_Linux.fmu
stop_time =  1
dt = 0.5

[Reinforcement Learning]
```