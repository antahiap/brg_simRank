import os
import copy
import pandas as pd

keyi = 'CCSA_sample.key'
data_path = '/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/CCSA_submodel/crash_modes'

s = pd.DataFrame()


ti = 1
dt = 0.2
dt_ui = 0.6

# flat mode
tL = []
for i in range(0, 6, 1):
    t = 1 + i*0.2
    tL.append(t)
tR = copy.copy(tL)

tL0, tR0 = copy.copy(tL), copy.copy(tR)
for i in range(1, 6, 1):
    dt = i*0.2

    for ts in tL0:
        # left stiff
        tL.append(ts + dt)
        tR.append(ts)

        # right stiff
        tL.append(ts)
        tR.append(ts + dt)

s['tL_u'] = tL
s['tR_u'] = tR
s['tL_i'] = [x + dt_ui for x in tL]
s['tR_i'] = [x + dt_ui for x in tR]

s['mL_u'] = ['2000121' for x in tL]
s['mR_u'] = ['2000121' for x in tL]
s['mL_i'] = ['2000142' for x in tL]
s['mR_i'] = ['2000142' for x in tL]

input(s)

for jId, r in s.iterrows():
    # update the sample key file
    with open(keyi) as kFile:
        kInp = kFile.read()

    print(r.tL_u, r.tR_u, r.tL_i, r.tR_i,
          r.mL_u, r.mR_u, r.mL_i, r.mR_i
          )
    keyUpdate = kInp.format(
        r.tL_u, r.tR_u, r.tL_i, r.tR_i,
        r.mL_u, r.mR_u, r.mL_i, r.mR_i
    )

    simName_f = 'CCSA_submodel_1{:03d}'
    simName = simName_f.format(jId)

    studPath = os.path.join(data_path, simName)
    if not os.path.exists(studPath):
        os.makedirs(studPath)

    outKey = simName + '.key'
    outKeyP = os.path.join(studPath, outKey)

    with open(outKeyP, "w") as f:
        f.write(keyUpdate)
        print(outKeyP)
