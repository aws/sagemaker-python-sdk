import json
import os

import horovod.tensorflow as hvd

if __name__ == '__main__':

    hvd.init()

    with open(os.path.join('/opt/ml/model/rank-%s' % hvd.rank()), 'w+') as f:
        basic_info = {'rank': hvd.rank(), 'size': hvd.size()}

        json.dump(basic_info, f)
        print('Saved file "rank-%s": %s' % (hvd.rank(), basic_info))

