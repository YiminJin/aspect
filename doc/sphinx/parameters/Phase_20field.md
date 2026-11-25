(parameters:Phase_20field)=
# Phase field


## **Subsection:** Phase field


::::{dropdown} __Parameter:__ {ref}`Crack geometry<parameters:Phase_20field/Crack_20geometry>`
:name: parameters:Phase_20field/Crack_20geometry
**Default value:**

**Pattern:** [List of <[Selection AT1|AT2|CZM ]> of length 0...1 (inclusive)]

**Documentation:** Crack-geometric law. The default is AT1.
::::

::::{dropdown} __Parameter:__ {ref}`Fracture energy<parameters:Phase_20field/Fracture_20energy>`
:name: parameters:Phase_20field/Fracture_20energy
**Default value:**

**Pattern:** [List of <[Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]> of length 0...1 (inclusive)]

**Documentation:** Positive fracture energy per unit crack area Gc in J/m^2. The default is 1.
::::

::::{dropdown} __Parameter:__ {ref}`Length scale<parameters:Phase_20field/Length_20scale>`
:name: parameters:Phase_20field/Length_20scale
**Default value:**

**Pattern:** [List of <[Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]> of length 0...1 (inclusive)]

**Documentation:** Positive regularization length in metres. The default is 1.
::::

::::{dropdown} __Parameter:__ {ref}`Phase composition<parameters:Phase_20field/Phase_20composition>`
:name: parameters:Phase_20field/Phase_20composition
**Default value:**

**Pattern:** [Anything]

**Documentation:** Name of the dimensionless phase field. The default is phase_field.
::::
