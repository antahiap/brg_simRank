import glob
import re

# pathdir = 'S:\\nobackup\\safety\\projects\\mma\\3_stv0\\front\\runs\\*'
pathdir = 'S:\\backup\\safety\\projects\\cx11*\\*\\front\\runs\\*\\*.key'
# pathdir = 'S:\\nobackup\\safety\\projects\\cx11*\\*\\front\\runs\\*\\binout0000'
lc_pose = 1

print(pathdir)
dirs = (glob.glob(pathdir, recursive = True))
lc_dic = {}
rl_dic = {}
print(len(dirs))
for d in dirs:
    # print(d)
    try:
        runName = d.split('\\')[-2]
        release = d.split('\\')[-5]
        loadcase = runName.split('_')[lc_pose]
    except IndexError:
        continue
    # print(runName, loadcase)
    # print(release)

    if loadcase in lc_dic.keys():
        lc_dic[loadcase] += 1
    else:
        lc_dic[loadcase] = 1

    if release in rl_dic.keys():
        rl_dic[release] += 1
    else:
        rl_dic[release] = 1
for k,v in lc_dic.items():
    print(k+ ':'+ str(v))

for k,v in rl_dic.items():
    print(k+ ':'+ str(v))
