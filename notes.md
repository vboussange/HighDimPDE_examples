# NOTES

## 20/04/2022
/!\ compare files on local and gpu server before commiting
### Goal
- have something descent with rep mut

### Problem
- The neural net does not converge to the initial solution, for d high
- this consequently triggers poor performance of the fit


### Plan for action
- [x] implement a sequential learning rate
- [ ] implement maxiters for each sequential learning rate.
- [ ] check on discourse whether such a thing as sequential learning rate is implemented. That might be indicated also in the doc of Flux.jl
- play around with the test file in folder `figure`
  - learning rate
  - shape of the nn
  - size of the domain

### opportunities
- the example allen cahn seems to work decently.