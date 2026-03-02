# End Effector Trajectory Controller

## Controller Overview

The end effector trajectory controller computes velocity commands to track a moving object's pose and twist. The controller uses a proportional control law with latency compensation to predict the object's future state.

## Notation

- $\mathbf{p}_{ee}(t)$: End effector position at time $t$
- $\mathbf{R}_{ee}(t)$: End effector rotation matrix at time $t$
- $\mathbf{v}_{ee}(t)$: End effector linear velocity at time $t$
- $\boldsymbol{\omega}_{ee}(t)$: End effector angular velocity at time $t$
- $\mathbf{p}_{obj}(t)$: Object position at time $t$
- $\mathbf{R}_{obj}(t)$: Object rotation matrix at time $t$
- $\mathbf{v}_{obj}(t)$: Object linear velocity at time $t$
- $\boldsymbol{\omega}_{obj}(t)$: Object angular velocity at time $t$
- $k_P^{pos}$: Position proportional gain
- $k_P^{rot}$: Rotation proportional gain
- $\tau$: System latency
- $\Delta t$: Simulation timestep

## Control Law

### Error Computation

The positional error is computed as:

$$\mathbf{e}_{pos} = \mathbf{p}_{obj} - \mathbf{p}_{ee}$$

The rotational error is computed in the end effector frame:

$$\mathbf{R}_{err} = \mathbf{R}_{ee}^T \mathbf{R}_{obj}$$

The rotational error vector $\mathbf{r}_{err}$ is extracted from $\mathbf{R}_{err}$ using the rotation vector representation:

$$\mathbf{r}_{err} = \text{rotvec}(\mathbf{R}_{err})$$

### Desired Velocity Computation

The desired linear velocity combines the object's velocity with a proportional correction term:

$$\mathbf{v}_{des}^{lin} = \mathbf{v}_{obj} + k_P^{pos} \mathbf{e}_{pos}$$

Similarly, the desired angular velocity is:

$$\boldsymbol{\omega}_{des} = \boldsymbol{\omega}_{obj} + k_P^{rot} \mathbf{r}_{err}$$

### Velocity Command

The velocity command (delta twist) is the difference between desired and current velocities:

$$\Delta \mathbf{v} = \mathbf{v}_{des}^{lin} - \mathbf{v}_{ee}$$

$$\Delta \boldsymbol{\omega} = \boldsymbol{\omega}_{des} - \boldsymbol{\omega}_{ee}$$

## Latency Compensation

To account for system latency $\tau$, the controller predicts the object's future pose:

$$\mathbf{p}_{obj}^{pred}(t + \tau) = \mathbf{p}_{obj}(t) + \mathbf{v}_{obj}(t) \cdot \tau$$

The rotation is integrated forward:

$$\mathbf{R}_{obj}^{pred}(t + \tau) = \mathbf{R}_{obj}(t) \cdot \exp(\boldsymbol{\omega}_{obj}(t) \times \tau)$$

where $\exp(\cdot)$ denotes the matrix exponential for rotation integration.

## Trajectory Simulation

The controller simulates the end effector trajectory over a horizon $T$ using the following algorithm:

1. Initialize: $\mathbf{p}_{ee}(0) = \mathbf{p}_{ee,0}$, $\mathbf{R}_{ee}(0) = \mathbf{R}_{ee,0}$, $\mathbf{v}_{ee}(0) = \mathbf{v}_{ee,0}$, $\boldsymbol{\omega}_{ee}(0) = \boldsymbol{\omega}_{ee,0}$

2. For each timestep $k = 1, 2, \ldots, N$ where $N = \lfloor T / \Delta t \rfloor$:
   1. Predict object pose at $t + \tau$:
      $$\mathbf{p}_{obj}^{pred} = \mathbf{p}_{obj}(k) + \mathbf{v}_{obj}(k) \cdot \tau$$
   
   2. Compute velocity command $\Delta \mathbf{v}(k)$, $\Delta \boldsymbol{\omega}(k)$ using the equations above
   
   3. Update end effector velocity:
      $$\mathbf{v}_{ee}(k+1) = \mathbf{v}_{ee}(k) + \Delta \mathbf{v}(k)$$
      $$\boldsymbol{\omega}_{ee}(k+1) = \boldsymbol{\omega}_{ee}(k) + \Delta \boldsymbol{\omega}(k)$$
   
   4. Integrate end effector pose:
      $$\mathbf{p}_{ee}(k+1) = \mathbf{p}_{ee}(k) + \mathbf{v}_{ee}(k+1) \cdot \Delta t$$
      $$\mathbf{R}_{ee}(k+1) = \mathbf{R}_{ee}(k) \cdot \exp(\boldsymbol{\omega}_{ee}(k+1) \times \Delta t)$$
   
   5. Integrate object pose (assuming constant velocity):
      $$\mathbf{p}_{obj}(k+1) = \mathbf{p}_{obj}(k) + \mathbf{v}_{obj}(k) \cdot \Delta t$$
      $$\mathbf{R}_{obj}(k+1) = \mathbf{R}_{obj}(k) \cdot \exp(\boldsymbol{\omega}_{obj}(k) \times \Delta t)$$

## Controller Parameters

The controller configuration includes:

- $k_P^{pos}$: Position proportional gain (units: s$^{-1}$)
- $k_P^{rot}$: Rotation proportional gain (units: s$^{-1}$)
- $\tau$: System latency (units: s)
- $T$: Simulation horizon (units: s)
- $\Delta t$: Timestep (units: s)

## Properties

- The controller uses feedforward from the object's velocity, enabling tracking of moving targets
- Latency compensation reduces tracking error for delayed systems
- The proportional control law ensures convergence to zero position and rotation error for static targets
- The controller is suitable for real-time trajectory generation and simulation


