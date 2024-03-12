---
layout: page
title: Metaclasses, Class for Classes
description: Python Metaclasses and Bayesian Statistics, an unexpected marriage
importance: 1
img: assets/img/inception.png
---

See [PR 158](https://github.com/aesara-devs/aeppl/pull/158) of AePPL.

In AePPL, variables from which we can obtain a log-probability function need to be instances of a `MeasurableVariable`. `RandomVariable`s are perhaps what comes first to mind when we think of object classes that inherit from `MeasurableVariable`, but other `Op`s also fall under this umbrella, e.g. `DiracDelta`, `MakeVector`, `CumOp`, etc. Note that the later two come under the form of `MeasurableMakeVector` and `MeasurableCumsum`, but that's a whole other story.

However, there are instances where we do _not_ want to extract the log-probability of a variable, notably components in a mixture model. The reason is because we just need the log-probability of the mixture and we "pull" the component-specific log-probability function using indexing. In AePPL mixtures, components are "turned into" instances of `UnmeasurableX`, where `X` is the original class of the variable/component. These classes are created _dynamically_, that is they are created "on the fly" when calling `assign_custom_measurable_outputs`. For instance, consider the following example:

```python
import aesara.tensor as at

X_rv = at.random.normal(5., 3., name="X")
Y_rv = at.random.normal(-5., 3., name="Y")
unmeasurable_X = assign_custom_measurable_outputs(X_rv.owner).op # <aesara.tensor.random.basic.UnmeasurableNormalRV at 0x1684808e0>
unmeasurable_Y = assign_custom_measurable_outputs(Y_rv.owner).op # <aesara.tensor.random.basic.UnmeasurableNormalRV at 0x168483550>
```

While `X_rv` and `Y_rv` are normal random variables, i.e. instances created from the same `NormalRV` class, `unmeasurable_X` and `unmeasurable_X` are their unmeasurable counterpart. However, while `assign_custom_measurable_outputs` dynamically creates `UnmeasurableNormalRV` at each call, these seemingly same object classes are **not** identical; notice already the difference memory addresses (0x1684808e0 and 0x168483550). This discrepancy was by design, i.e. they could have been the same class but we chose not.

However, the creation of new classes every time posed the issue of having each `UnmeasurableNormalRV` to not be "equal", which we would want. Here, the use of language can be confusing; we want each `UnmeasurableNormalRV` classes to have the _same_ hash but _different_ ids. Having the same hash allows the ops to be "equal" by Python standards.

```python
hash(unmeasurable_X) == hash(unmeasurable_Y) # True: 4967640381975027986 == 4967640381975027986
id(unmeasurable_X) == id(unmeasurable_Y) # False: 6044493248 == 6044530000

unmeasurable_X == unmeasurable_Y # True, same hashes
unmeasurable_X is unmeasurable_Y # False, different ids
```

To allow object classes to be dynamically created yet have the same hash, we resort to [metaclasses](https://en.wikipedia.org/wiki/Metaclass), a class whose instances are classes themselves. The example above stems from the use of a newly created Python metaclass which I called `UnmeasurableMeta`, which inherits from `aesara.graph.utils.MetaType` which itself inherits from `abc.ABCMeta`, an Inception-like concept and the supreme lord of Python metaclasses.

```python
class UnmeasurableMeta(MetaType):
    def __new__(cls, name, bases, dict):
        if "id_obj" not in dict:
            dict["id_obj"] = None

        return super().__new__(cls, name, bases, dict)

    def __eq__(self, other):
        if isinstance(other, UnmeasurableMeta):
            return hash(self.id_obj) == hash(other.id_obj)
        return False

    def __hash__(self):
        return hash(self.id_obj)

class UnmeasurableVariable(metaclass=UnmeasurableMeta):
    """
    id_obj is an attribute, i.e. tuple of length two, of the unmeasurable class object.
    e.g. id_obj = (NormalRV, noop_measurable_outputs_fn)
    """
```

Effectively, the dunder methods `__eq__` and `__hash__` are overwritten to allow the behaviour above. For more information on the behaviour of unmeasurable variables, please see the newly added tests in `tests/test_abstract.py` of the AePPL repository.
