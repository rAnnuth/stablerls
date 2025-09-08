# Contribution
We have a strong background in electrical engineering; therefore, this is the field of research to which we apply this tool. However, we are convinced that there are more subjects for which this tool can be used. In general, this package can be applied to any MATLAB Simulink simulation. If you have any suggestions about improving the package or want to provide additional examples of how you are using this tool, we are happy to work together. Since, Reinforcement Learning has always had many problem specific parts, we decided a fork-and-branch git workflow is the best approach ([Guide](https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/)).

## Creating Issues
The issue should briefly point out the issue and contain all relevant information for others to understand the problem or your idea to improve the package. The best case is to include a minimal working example. Please use the following labels to mark your issue:

`bug`, `feature`, `help`

## Development Workflow
The guide mentioned above contains all relevant information about the workflow, however, the main steps are summarized below:

1. Fork this repository after you log in
2. Clone the forked repository. The link should contain your username
3. Develop and change the code 
4. To contribute your work change the remote repository `git remote add upstream https://github.com/rAnnuth/stablerls.git` 
5. Create a new branch and push your code
6. Check if everything still works as expected by running the Pytests in the ./Test folder
7. Create a pull request where you document your changes