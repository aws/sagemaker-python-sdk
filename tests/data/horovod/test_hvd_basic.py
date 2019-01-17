import json
import os
import horovod.tensorflow as hvd

hvd.init()

with open(os.path.join('/opt/ml/model/rank-%s' % hvd.rank()), 'w+') as f:
    basic_info = {'rank': hvd.rank(), 'size': hvd.size()}

    print(basic_info)
    json.dump(basic_info, f)
