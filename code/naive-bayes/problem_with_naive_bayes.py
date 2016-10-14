__author__ = 'theopavlakou'

from models.Mixture_of_Gaussians import Mixture_of_Gaussians as MoGs
from plotting.plotting_helper import plot_contour_quantiles, plot_contour
import seaborn as sns
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
import numpy as np

levels = [0.1, 0.05, 0.025]

x_sym = T.matrix('x_sym')
mus_class_1 = np.array([[-1, 3]])
Sigmas_class_1 = np.array([[[1, 0.8],
                    [0.8, 1]]])
p = np.array([1])

mog_real_class_1 = MoGs(x_sym, 2, 1, mu=mus_class_1, Sigma=Sigmas_class_1, p=p)
p_x_mog_real_class_1 = T.exp(mog_real_class_1.get_symbolic_log_prob_per_data_point())
p_x_mog_real_class_1_function = theano.function([x_sym], p_x_mog_real_class_1, allow_input_downcast=True)

mus_class_2 = np.array([[2, 0]])
Sigmas_class_2 = np.array([[[1, -0.8],
                    [-0.8, 1]]])

mog_real_class_2 = MoGs(x_sym, 2, 1, mu=mus_class_2, Sigma=Sigmas_class_2, p=p)
p_x_mog_real_class_2 = T.exp(mog_real_class_2.get_symbolic_log_prob_per_data_point())
p_x_mog_real_class_2_function = theano.function([x_sym], p_x_mog_real_class_2, allow_input_downcast=True)

Sigmas_class_1_estimated = np.array([[[1, 0.0],
                                [0.0, 1]]])

mog_class_1_estimated = MoGs(x_sym, 2, 4, mu=mus_class_1, Sigma=Sigmas_class_1_estimated, p=p)
p_x_mog_class_1_estimated = T.exp(mog_class_1_estimated.get_symbolic_log_prob_per_data_point())
p_x_mog_class_1_estimated_function = theano.function([x_sym], p_x_mog_class_1_estimated, allow_input_downcast=True)


Sigmas_class_2_estimated = np.array([[[1, 0.0],
                                [0.0, 1]]])

mog_class_2_estimated = MoGs(x_sym, 2, 4, mu=mus_class_2, Sigma=Sigmas_class_2_estimated, p=p)
p_x_mog_class_2_estimated = T.exp(mog_class_2_estimated.get_symbolic_log_prob_per_data_point())
p_x_mog_class_2_estimated_function = theano.function([x_sym], p_x_mog_class_2_estimated, allow_input_downcast=True)

# sns.set_style("dark", rc={"lines.linewidth": 2.5, "axes.facecolor": "#f6f0ff"})

f, axarr = plt.subplots(1, 2)
f.set_size_inches(18.5, 10.5, forward=True)

cs_1 = plot_contour(-4, 5, -4, 6, axarr[0], p_x_mog_real_class_1_function, use_contourf=False, levels=levels)
# f.colorbar(cs_1, ax=axarr[0][0])
axarr[0].set_title("Actual class conditional Gaussians")

cs_2 = plot_contour(-4, 5, -4, 6, axarr[0], p_x_mog_real_class_2_function, use_contourf=False, levels=levels)
# f.colorbar(cs_1, ax=axarr[0][0])
# axarr[0, 1].set_title("Probability density for original MoG")
axarr[0].scatter(np.array([0]), np.array([2]), c=sns.cubehelix_palette(8)[6])

cs_3 = plot_contour(-4, 5, -4, 6, axarr[1], p_x_mog_class_1_estimated_function, use_contourf=False, levels=levels)
# f.colorbar(cs_1, ax=axarr[0][0])
axarr[1].set_title("Estimated class conditional Gaussians")

cs_4 = plot_contour(-4, 5, -4, 6, axarr[1], p_x_mog_class_2_estimated_function, use_contourf=False, levels=levels)
# f.colorbar(cs_1, ax=axarr[0][0])
# axarr[0, 1].set_title("Probability density for original MoG")
axarr[1].scatter(np.array([0]), np.array([2]), c=sns.cubehelix_palette(8)[6])



# f.colorbar(cs_4, ax=axarr[1][1])
# axarr[1, 1].set_title("Probability density for RNADE trained on rotated data")

# f.savefig("images/toy-data/rnade-mogs-rotated-demo-contour-plot.jpg")
plt.show()