# PYTHON script
import oems


def pidList(oem):
    oem.set_data_path()
    oem.backend_server()
    driver = oem.driver
    cypherTxt = '''MATCH (n:Part) RETURN distinct n.part_id'''

    results = driver.session().run(cypherTxt)
    pidList = [r['n.part_id'] for r in results]
    print(pidList)


# oem = oems.oems('YARIS')

def main ():
    runSet = [
        # ['3_stv0', ['fp3', 'fo5', 'fod']],
        # ['2_stcr', ['fp3', 'fo5', 'fod']],
        ['3_m1', ['fp3', 'fo5', 'fod']],
        ['3_stv02', ['fp3', 'fo5', 'fod']],
        ['3_stv03', ['fp3', 'fo5', 'fod']],
        ]
    oem_name = 'CEVT'

    for run in runSet:
        print(run)
        rls = run[0].split('_')[-1]
        for lc in run[1]:
            oem = oems.oems(oem_name, rls, lc)
            oem.set_data_path()
            oem.metapost()
    

def update_pidList():
# meta can't have neo4j driver, need to run this before runing meta -s metapost.py
    runSet = [
        # ['3_stv0', ['fp3', 'fo5', 'fod']],
        # ['2_stcr', ['fp3', 'fo5', 'fod']],
        ['3_m1', ['fp3', 'fo5', 'fod']],
        ['3_stv02', ['fp3', 'fo5', 'fod']],
        # ['3_stv03', ['fp3', 'fo5', 'fod']],
        ]
    oem_name = 'CEVT'

    oem = oems.oems(oem_name)
    cy = oem.cypher()
    for run in runSet:
        rls = run[0].split('_')[-1]
        for lc in run[1]:
            pids_out = [ str(x) for x in cy.make_pidList('.*{0}.*{1}.*'.format(rls, lc))]
            pids_out = ', '.join(pids_out)

            with open('pids.py', 'a') as f:
               f.write('{0}_{1} = {2}\n'.format(rls, lc, repr(pids_out)))
    
    
if __name__ == '__main__':

    # update_pidList()
    main()
