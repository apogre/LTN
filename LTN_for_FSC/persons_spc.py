import logictensornetworks as ltn
import numpy as np

number_of_features = 100
#identity matrix 30 * 30
features_for_generic_peoples = np.identity(number_of_features).tolist()

person = ltn.Domain_union([ltn.Constant("p"+str(i), features_for_generic_peoples[i]) for i in range(number_of_features)])

everybody = []

spouse_set = []
parents_set = []
children_set = []
