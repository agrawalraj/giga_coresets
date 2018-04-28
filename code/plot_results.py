
import numpy as np
import matplotlib.pyplot as plt
from utilities import make_clean_dict, plot_results_rfm

human_class = make_clean_dict('../results/human_activity', 2)
mnist_class = make_clean_dict('../results/mnist', 2)
sensor_class = make_clean_dict('../results/sensorless', 2)
adult_class = make_clean_dict('../results/adult', 2)

human_kern = make_clean_dict('../results/human_activity', 3)
mnist_kern = make_clean_dict('../results/mnist', 3)
sensor_kern = make_clean_dict('../results/sensorless', 3)
adult_kern = make_clean_dict('../results/adult', 3)

# Plot classification accuracies
f, axs = plt.subplots(nrows=2, ncols=2)
f.set_figheight(20)
f.set_figwidth(20)
plot_results_rfm(human_class, axs[0, 0], title='Human Activity')
plot_results_rfm(mnist_class, axs[0, 1], title='MNIST')
plot_results_rfm(sensor_class, axs[1, 0], title='Sensorless')
plot_results_rfm(adult_class, axs[1, 1], title='Adult')

# Plot relative matrix approximation errors
f, axs = plt.subplots(nrows=2, ncols=2)
f.set_figheight(20)
f.set_figwidth(20)
plot_results_rfm(human_kern, axs[0, 0], 3, 'Human Activity')
plot_results_rfm(mnist_kern, axs[0, 1], 3, 'MNIST')
plot_results_rfm(sensor_kern, axs[1, 0], 3, 'Sensorless')
plot_results_rfm(adult_kern, axs[1, 1], 3, 'Adult')

pickle.load( open('../results/human_activity' + '/' + 'human_activity_5k_20k_iter_0.p', "rb" ) )

