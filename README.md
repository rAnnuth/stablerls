![](icon.png)

<h2 align="center">Stable Reinforcement Learning for Simulink</h2>

<p align="center">
<a href="https://stablerls.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/stablerls/badge/?version=latest"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
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
Action Status, Coverage Status
sphinx-autobuild docs/ docs/build/html

[project.optional-dependencies]
doc = [
    "sphinx ~=4.5.0",
    "myst-parser",
    "furo",
]
```