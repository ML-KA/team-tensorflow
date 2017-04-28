import itertools
import json
import subprocess

learning_rate = [0.01, 0.1, 1]
batch_size = [128, 512, 2048, 4096]
n_epochs = [10, 50, 100]
optimizer = ['gradient_descent', 'adadelta', 'adam']

config_list = list(itertools.product(learning_rate,
                                     batch_size,
                                     n_epochs,
                                     optimizer))

no_of_configs = len(config_list)

list_of_dicts = []

for config in config_list:
    config_dict = {
        "learning_rate": config[0],
        "batch_size": config[1],
        "n_epochs": config[2],
        "optimizer": {
            "gradient_descent": config[3] == "gradient_descent",
            "adadelta": config[3] == "adadelta",
            "adam": config[3] == "adam"
        }
    }

    list_of_dicts.append(config_dict)
# print(list_of_dicts[0:10])

counter = 0
for config_dict in list_of_dicts:
    counter = counter + 1
    with open('config.json', 'w') as config_file:
        json.dump(config_dict, config_file)

    p = subprocess.call(['python',
                         'logistic_regression_not_mnist.py'])

    print('Did {0} from {1} configs'.format(counter, no_of_configs))
    print('Configuration: {0}'.format(config_dict))
