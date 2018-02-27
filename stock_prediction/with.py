def hoge():
    wfp.write('hgoehoge')
    print('hoge is called')

with open('msg.log', 'w') as wfp:
    wfp.write('it is example for with statement.')
    wfp.write('do not need call close().')

    hoge()
