import glob
import re

# pathdir = 'S:\\nobackup\\safety\\projects\\mma\\3_stv0\\front\\runs\\*'
pathdir = 'S:\\nobackup\\safety\\projects\\mma\\2_stcr\\front\\runs\\*'
# pathdir = '/cevt/cae/backup/safety/users/anahita.pakiman1/mma/*/rear/runs/*'  # 1
lc_pose = 3 #front, rear

lc_pose = 1 # side
pathdir = '/cevt/cae/backup/safety/users/anahita.pakiman1/mma/*/side/runs/*'  # 1

print(pathdir)
dirs = (glob.glob(pathdir))
lc_dic = {}
print(len(dirs))
for d in dirs:
    print(d)
    try:
        # runName = d.split('\\')[-1]
        runName = d.split('/')[-1]
        loadcase = runName.split('_')[lc_pose]
        rls = d.split('/')[-4]
    except IndexError:
        continue

    if loadcase in lc_dic.keys():
        lc_dic[loadcase] += 1
    else:
        lc_dic[loadcase] = 1


for k,v in lc_dic.items():
    print(k+ ':'+ str(v))
