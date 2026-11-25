(sec:methods:compositional-fields)=
# Compositional fields

The last of the basic equations, {math:numref}`eq:compositional`, describes the evolution of a set of variables $c_i(\mathbf x, t), i=1\ldots C$ that we typically call *compositional fields* and that we often aggregate into a vector $\mathfrak c$.

Compositional fields were originally intended to track what their name suggest, namely the chemical composition of the convecting medium. In this interpretation, the composition is a non-diffusive quantity that is simply advected along passively, i.e., it would satisfy the equation
```{math}
\begin{aligned}
  \frac{\partial \mathfrak c}{\partial t} + \mathbf u \cdot \nabla \mathfrak c = 0.
\end{aligned}
```
However, the compositional fields may also participate in determining the values of the various coefficients as discussed in {ref}`sec:methods:coefficients`, and in this sense the equation above describes a composition that is *passively advected*, but an *active participant* in the equations.

That said, over time compositional fields have shown to be a much more useful tool than originally intended.
For example, they can be used to track where material comes from and goes to (see {ref}`sec:cookbooks-composition`) and, if one allows for a reaction rate $\mathfrak q$ on the right hand side,
```{math}
\begin{aligned}
  \frac{\partial \mathfrak c}{\partial t} + \mathbf u \cdot \nabla \mathfrak c = \mathfrak q,
\end{aligned}
```
then one can also model interaction between species - for example to simulate phase changes where one compositional field, indicating a particular phase, transforms into another phase depending on pressure and temperature, or where several phases combine to other phases.
Another example of using a right hand side - quite outside what the original term *compositional field* was supposed to indicate - is to track the accumulation of finite strain, see {ref}`sec:cookbooks:finite_strain`.

In actual practice, one finds that it is often useful to allow $\mathfrak q$ to be a function that has both a smooth (say, continuous) in time component, and one that is singular in time (i.e., contains Dirac delta, or "impulse" functions).
Typical time integrators require the evaluation of the right hand side at specific points in time, but this would preclude the use of delta functions.
Consequently, the integrators in ASPECT only require material models to provide an *integrated* value $\int_t^{t+\Delta t} \mathfrak q(\tau) \; \text{d}\tau$ through the `reaction_term` output variable.
Implementations often approximate this as $\triangle t \cdot \mathfrak q(t)$, or similar formulas.

A second application for only providing integrated right hand sides comes from the fact that modeling reactions between different compositional fields often involves finding an equilibrium state between different fields because chemical reactions happen on a much faster time scale than transport.
In other words, one then often assumes that there is a $\mathfrak c^\ast(p,T)$ so that
```{math}
\begin{aligned}
  \mathfrak q(p,T,\varepsilon(\mathbf u),\mathfrak c^\ast(p,T)) = 0.
\end{aligned}
```
Consequently, the material model methods that deal with source terms for the compositional fields need to compute an *increment* $\Delta\mathfrak c$ to the previous value of the compositional fields so that the sum of the previous values and the increment equals $\mathfrak c^\ast$.
This corresponds to an *impulse change* in the compositions at every time step, as opposed to the usual approach of evaluating the right hand side term $\mathfrak q$ as a continuous function in time, which corresponds to a *rate*.

On the other hand, there are other uses of compositional fields that do not actually have anything to do with quantities that can be considered related to compositions.
For example, one may define a field that tracks the grain size of rocks.
If the strain rate is high, then the grain size decreases as the rocks break.
If the temperature is high enough, then grains heal and their size increases again.
Such "damage models" would then introduce a quantity $c(t)$ describing the "damage" to the material (here assumed to be described by a single scalar field) that satisfies an equation of the form
```{math}
\begin{aligned}
  \frac{\partial c}{\partial t} + \mathbf u \cdot \nabla c = q(T,c),
\end{aligned}
```
where in the simplest case (much simplified from real models) one could postulate
```{math}
\begin{aligned}
  q(T,c) =  A \dot\varepsilon - B \max\{T-T_{\text{healing}},0\} c.
\end{aligned}
```
Here, $\dot\varepsilon$ is the strain rate that causes damage; the first term then leads to growth of damage as strain continues to accumulate on the material.
The second term *decreases* the damage if the temperature is high enough.
One would then use this compositional field in the definition of the viscosity of the material: more damage means lower viscosity because the rocks are weaker.

In cases like this, there is only a single compositional field and it is not in permanent equilibrium.
Consequently, the increment implementations of material models in ASPECT need to compute is typically the rate $q(T,c)$ times the time step.
In other words, if you compute a reaction rate inside the material model you need to multiply it by the time step size before returning the value.

Compositional fields have proven to be surprisingly versatile tools to model all sorts of components of models that go beyond the simple Stokes plus temperature set of equations.
Play with them!

:::{note}
As has hopefully become clear from the discussion above, the term "compositional field" as used in ASPECT is by now mostly historic: These fields were meant to track chemical compositions, but are now used to track all sorts of other things as well, or in some cases track nothing at all and just be static fields that simply indicate where some features of the model are located. It is therefore useful to think of the term "compositional field" as a *technical term* in which the two words appear together, separate from the original meaning of the word "compositional".
:::

## Selecting an explicit advection method

Each entry in `Compositional field methods` applies to the corresponding field name.
For example, this input advances the companion with the ordinary implicit method
and the tracer with third-order strong-stability-preserving Runge–Kutta (SSPRK3).

```text
subsection Mesh refinement
  set Time steps between mesh refinement = 0
end
subsection Discretization
  set Composition polynomial degree = 2
  set Use discontinuous composition discretization = true
end
subsection Compositional fields
  set Number of fields = 2
  set Names of fields = companion, tracer
  set Compositional field methods = field, explicit DG SSPRK3
end
```

The explicit method transports a DGQ2 scalar with the accepted Q2 velocity on a fixed,
conforming, nonperiodic Cartesian box in 2-D or 3-D. Generic and chemical tracers
use signed transport. A field of type `phase field` uses bounded quadratic Bernstein
transport. Prescribed inflow uses the Cartesian boundary composition function at
each stage time. Use a single-advection scheme with prescribed, single or iterated
Stokes velocity. Explicit fields require pure advection with zero reaction increments,
no prescribed DoFs and the ordinary composition limiter disabled. Operator splitting,
melt transport and ASPECT elasticity are unsupported in this configuration.

An empty method list defaults to `field`. A single entry applies to every composition.
Existing inputs using `SSPRK3 fields = tracer` select the same explicit method
after this expansion. A name may occur once in that list and its method must be
either `field` or `explicit DG SSPRK3`.

For the growing fault, select `midpoint fault phase` on one discontinuous quadratic
field of type `phase field`. The coupled mechanics and irreversible phase solve
advances this field in the single-advection, iterated Newton Stokes scheme.
Configure AT1 on the same named composition under
the {ref}`Phase field <parameters:Phase_20field>` subsection. Older inputs may use
`Postprocess / Phase field surface energy` for the same four settings. Settings
provided in both locations must agree.
These parameters also select the functional for the optional energy postprocessor.
With an ordinary composition and no fault material, the postprocessor can measure
AT1, AT2 or CZM surface energy in two or three dimensions.
