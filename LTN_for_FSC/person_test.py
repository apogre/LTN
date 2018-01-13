import logictensornetworks as ltn
import numpy as np

number_of_features = 100

#identity matrix 30 * 30
features_for_generic_peoples = np.identity(number_of_features).tolist()

person = ltn.Domain_union([ltn.Constant("p"+str(i), features_for_generic_peoples[i]) for i in range(number_of_features)])


barack = ltn.Constant("Barack", domain=person)
michelle = ltn.Constant("Michelle", domain=person)
sasha = ltn.Constant("Sasha", domain=person)
malia = ltn.Constant("Malia", domain=person)

#Group 2
bill = ltn.Constant("Bill", domain=person)
hillary = ltn.Constant("Hillary", domain=person)
chelsea = ltn.Constant("Chelsea", domain=person)