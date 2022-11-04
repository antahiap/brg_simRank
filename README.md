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

- [x] Pe plot with selecting only closest

## update cevt result

- [x] run the result
- [x] change the method
- [x] verify the result
  - [x] whey there is similarities close to zero, is it 0 or sth else> tehre are smaller than 0.11 but not zero
  - [x] why hh-ll doen't pick small HH anymore > errList is active
  - [x] check the KDE if the smlll similarities belong to pid than sims > the errList was empty

### scatter plot comparison

- [x] the color code, L sim to be red
- [x] legend order
- [x] save four plot
- [x] add four pic in tex
- [x] legend inside

- [x] fix the bolt for plot_report.py
- [x] section 6.2, 4 pid -> components
- [x] split the paper
- [ ] cite the conf paper modeling

### Graph visualization

- [x] make same vis as paper
  - [x] check the error list > it was ok, the issue was that initital plot was without weight - one des is missing in curre
- [x] show loop with removing high distance nodes
  - [x] why network2tikzi doesn't plot > issue with latex, set the style with plotly
- [ ] show visualization with more releases / several rls together
