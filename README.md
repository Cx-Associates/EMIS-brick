# EMIS-brick
CxA's EMIS for brick data models

Typical workflow: you want to create, train, test, or run performance reports on a number of energy models

An instance of EnergyModelset is primarily a set of energy models, but it also provides context to store project-level objects such as a single performance period used to report across multiple models, the project instance itself (and therefore attributes such as lat, long location, which is used to retrieve weather data), and other useful objects that are used  . Its class is defined in the `EMIS-brick` code becaues it repurposes elements of both submodules (described below). The EnergyModelset class provides context to store project-level objects like 

## The submodules: brickwork and energy_models
* `brickwork` uses the Brick Schema to organize building metadata. In order to run code in `EMIS-brick`, you must have at least a rudimentary brick model in .ttl format. When you run a function like `energy_modelset.get_data()`, the code uses 
* `energy_models` is a repository of analytics code used for energy modeling in the style of ASHRAE 14, U.S. Superior Energy Performance, "Strategic Energy Management" (ISO 500001), performance contracting, and other modeling initiatives that use the conventions of baseline and performance period to measure energy savings.

hard-coded aspects:
* a `Modelset` object can store models within one of two attributes only: `equipment` or `systems`
* UNIQUENESS.


## Time Frames

Time frames can be properties of various classes. This complexity introduces sources of error; for example, there may be a function that calls for the time frame of a particular piece of equipment or a system, and if the time-frame has only been set at the project level and not the equipment or system in that case, the code will throw an error. Regardless, the complexity of being able to assign time-frames at different levels (from project down to individual energy models) is warranted because within one project, energy models may have different time periods.

In training a set of new energy models, it makes sense for the energy models to inherit the baseline timeframe from the project level to begin with. But over time, you may need to adjust the baseline period of individual models based on data availability or the favorability of the training results. It is common to look for consistent baseline periods that allow the model to converge on favorable metrics. So in this context, the ability to set individual baseline time_frames for each model is desirable. There may be a more clever way to implement time frame instantiation and inheritence than the way it's currently handled in the code base.

For each 'job,' it's expected that the reporting periods will stay the same. I.e., even if you have models with different baseline periods, they will probably all she the same reporting periods for a given application / a given reporting run.