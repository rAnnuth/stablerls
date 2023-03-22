# StableRLS 
[![Documentation Status](https://readthedocs.org/projects/stablerls/badge/?version=latest)](https://stablerls.readthedocs.io/en/latest/?badge=latest)

####Stable Reinforcement Learning for Simulink
Description ... and probably logo

## Structure 
- Examples: Usage examples of the package
- StableRLS: Main package
- Test: Verify functionality

## TODO
- [ ] Add sphinx version to package

**Step 1**:
- [x] Find a name
- [x] Move code to other repository
- [ ] Define test cases
- [x] Go Gymnasium
- [ ] Use Loggers

**Step 2**:
- [ ] Document code and check for errors
- [ ] Create at least two test cases

**Step 3**:
- [ ] Move to Github
- [ ] Write paper and make ready for submission 

**Step 4**:
- [ ] [https://joss.theoj.org/about](https://joss.theoj.org/about) Submit


```
sphinx-autobuild docs/ docs/build/html

[project.optional-dependencies]
doc = [
    "sphinx ~=4.5.0",
    "myst-parser",
    "furo",
]
```