# Revision modifications

## Graound truth for the simrank

- [x] read d3plot displacements
- [x] one time step, last

  - [x] calcualte the diff
  - [x] all time steps
  - [x] just 5 parts

- [x] Make the todo tasks
- [x] Compare 3001
  - [x] Simrank matrix value
    - [x] argsort issue with small value, `numpy.array([1., 0.5, 1.]).argsort(kind="heapsort")`
  - [x] Simrank loop
  - [x] make feature graph
  - [x] Pe vs IE weight
    - [x] why from single to all paired sim differ
    - [x] Have two different weight
    - [ ] Correct kg01 loading
- [x] Similarity from RMSE
- [ ] How is it with component graph
- [x] Diagnoal value not 1 > effect of spread
  - [x] Look in paper
  - [ ] Move the test functions
- [x] Run simrank for each simulation and ref seprately
- [x] What does SimRank do
- [ ] Change the spread function

## Extended data debug

- get top color
- decrease number of simulation range

## Write result

- [x] update displacements
- [ ] update text
- [ ] write extended result

## SimRank issue

- spread normalization differ from simrank weighted normalization. spread normalize vs source edges where simRank normalizes vs target edges
- [x] recalculate spread
- [x] why sim > 1
  - [x] check sim for test_epread > not bigger than 1
  - [x] check combining sprd and evd
- [x] change simRankpp function to simrank_pp_similarity_numpy in nrg_simRank/inv_simRnk_extnd.py
- [x] clean sprd, evd, C option in nrg_simRank/inv_simRnk_extnd.py
- [x] run the extnd result
- [x] check simrank result with github functions > wrong result on the simple example
- [x] transfer result to the plot_nrg_extnd

- [x] spread only for PID nodes
- [ ] change weight manipulation to weight subtraction

## Make new result
- [ ] IE plot with selecting only closest
