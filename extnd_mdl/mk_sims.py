import os
import copy
import pandas as pd

keyi = 'CCSA_sample.key'
data_path = '/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/CCSA_submodel/crash_modes'

s = pd.DataFrame()


ti = 1
dt = 0.2
dt_ui = 0.6
id = '1'
simName_f = 'CCSA_submodel_'

# set 1 dt_tu= 0.6  wrong setup, ti > tu, overwriten pkl
# set 2 dt_tu=0.4   wrong setup, ti > tu, overwriten pkl

# set 3  dt_tu= 0.6 dt =2        tu > ti
# set 4  dt_tu= 0.4 dt =2        tu > ti
# set 5, dt=0.05, dt_ui=0.6      tu > ti
# set 6, dt=1, dt_ui=0.6         tu > ti

# flat mode
j, sims_id = int(id) * 1000, []
tL = []
for i in range(0, 6, 1):
    t = 1 + i*dt
    tL.append(t)
    sims_id.append(j)
    j += 1
tR = copy.copy(tL)


tL0, tR0 = copy.copy(tL), copy.copy(tR)
for i in range(1, 6, 1):
    dti = i*dt

    for ts in tL0:
        sims_id.append(j)
        # left stiff
        tL.append(ts + dti)
        tR.append(ts)

        sims_id.append(j+1)
        # right stiff
        tL.append(ts)
        tR.append(ts + dti)
        j += 2

s['tL_i'] = tL
s['tR_i'] = tR
s['tL_u'] = [x + dt_ui for x in tL]
s['tR_u'] = [x + dt_ui for x in tR]

s['mL_u'] = ['2000121' for x in tL]
s['mR_u'] = ['2000121' for x in tL]
s['mL_i'] = ['2000142' for x in tL]
s['mR_i'] = ['2000142' for x in tL]
s['dt_ui'] = str(dt_ui)
s['id'] = sims_id

s['name'] = [simName_f + str(ids) for ids in sims_id]

s.to_pickle("./sims_extnd_0{}.pkl".format(id))
input(s)

for jId, r in s.iterrows():
    # update the sample key file
    with open(keyi) as kFile:
        kInp = kFile.read()

    print(r.tL_u, r.tR_u, r.tL_i, r.tR_i,
          r.mL_u, r.mR_u, r.mL_i, r.mR_i
          )

    ids = '{}{:03d}'.format(id, jId)
    simName = simName_f + ids

    keyUpdate = kInp.format(
        r.tL_u, r.tR_u, r.tL_i, r.tR_i,
        r.mL_u, r.mR_u, r.mL_i, r.mR_i,
        simName
    )

    studPath = os.path.join(data_path, simName)
    if not os.path.exists(studPath):
        os.makedirs(studPath)

    outKey = simName + '.key'
    outKeyP = os.path.join(studPath, outKey)
    print(outKeyP)

    with open(outKeyP, "w") as f:
        f.write(keyUpdate)
        print(outKeyP)
