#!/usr/bin/env python
__author__ = "Luciano Serafini"
__copyright__ = "Copyright 2017, Fondazione Bruno Kessler"
__email__ = "serafini@fbk.eu"

import tensorflow as tf
import logictensornetworks as ltn
import numpy as np
import os
import csv

ltn.default_layers = 10
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "luk"
ltn.default_aggregator ="hmean"
ltn.default_optimizer = "rmsprop"
ltn.default_clauses_aggregator = "hmean"
ltn.default_fact_penality = 1e-6

number_of_features = 30

#identity matrix 30 * 30
features_for_generic_peoples = np.identity(number_of_features).tolist()

person = ltn.Domain_union([ltn.Constant("p"+str(i),features_for_generic_peoples[i]) for i in range(number_of_features)])


#Group 1
barack = ltn.Constant("Barack",    domain=person)
michelle = ltn.Constant("Michelle",      domain=person)
sasha = ltn.Constant("Sasha",  domain=person)
malia = ltn.Constant("Malia",domain=person)

#Group 2
bill = ltn.Constant("Bill", domain=person)
hillary = ltn.Constant("Hillary", domain=person)
chelsea = ltn.Constant("Chelsea", domain=person)

# {'columns': 3,
#  'label': 'Barack',
#  'parameters': [<tf.Variable 'Barack:0' shape=(1, 3) dtype=float32_ref>],
#  'tensor': <tf.Variable 'Anna:0' shape=(1, 3) dtype=float32_ref>}


#child knowledge is available
first_group = [barack,michelle,sasha,malia]

#child knowledge is not available
second_group = [bill,hillary,chelsea]

everybody = first_group + second_group

person = ltn.Domain_union(everybody)

p1p1 = ltn.Domain_concat([person, person])

p1p2 = ltn.Domain_product(person, person)

p1 = ltn.Domain_slice(p1p2, 0, number_of_features)

p2 = ltn.Domain_slice(p1p2, number_of_features, number_of_features*2)

p2p1 = ltn.Domain_concat([p2, p1])
#reversing p1p2, each of individual persons
#
# some_friend = ltn.Function("some_friend", person, person)
#
# some_friend_of_person = ltn.Term(some_friend, person)
#
#
Are_spouse = ltn.Predicate("Are_spouse", ltn.Domain_concat([person, person]))
Are_parent = ltn.Predicate("Are_parent", ltn.Domain_concat([person, person]))
Are_child = ltn.Predicate("Are_child", ltn.Domain_concat([person, person]))

#
# # Smokes(person)
# Smokes = ltn.Predicate("Smokes", person)
#
# Has_cancer = ltn.Predicate("has_cancer", person)
#
# smokes_implies_has_cancer = \
#     [ltn.Clause([ltn.Literal(False, Smokes, person),
#                  ltn.Literal(True, Has_cancer, person)], label="smoking_implies_cancer")]
#
# friends_xy_imp_smokes_x_imp_smokes_y = \
#     [ltn.Clause([ltn.Literal(False, Are_friends, p1p2),
#                  ltn.Literal(False, Smokes, p1),
#                  ltn.Literal(True, Smokes, p2)],
#                 label="friends_p1p2_and_smokes_p1_implies_smokes_p2")]
#
spouse_is_symmetric = \
    [ltn.Clause([ltn.Literal(False, Are_spouse, p1p2),
                 ltn.Literal(True, Are_spouse, p2p1)],
                label="spouse_p1p2_implies_spouse_p2p1")]

child_p1p2_implies_parent_p2p1 = \
    [ltn.Clause([ltn.Literal(False, Are_child, p1p2),
                 ltn.Literal(True, Are_parent, p2p1)],
                label="child_p1p2_implies_parent_p2p1")]
#
# everybody_has_a_friend = [ltn.Clause([ltn.Literal(True, Are_friends, ltn.Domain_concat([person, some_friend_of_person]))],
#                                      label="everybody_has_a_friend")]


spouse_of = {barack:{michelle},
             michelle:set(),
             sasha:set(),
             malia:set(),
             bill:{hillary},
             hillary:set(),
             chelsea:set()}


spouse_positive_examples = [ltn.Clause([ltn.Literal(True, Are_spouse, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_and_"+y.label+"_are_spouse") for x in spouse_of for y in spouse_of[x]]


spouse_negative_examples =  [ltn.Clause([ltn.Literal(False, Are_spouse, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_and_"+y.label+"_are_not_spouse")
                             for group in [first_group, second_group]
                             for x in group for y in group
                             if y not in spouse_of[x] and x not in spouse_of[y]]


child_of = {barack:set(),
             michelle:set(),
             sasha:{barack,michelle},
             malia:{michelle,barack},
             bill:set(),
             hillary:set(),
             chelsea:{bill,hillary}}


child_positive_examples = [ltn.Clause([ltn.Literal(True, Are_child, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_child_of_"+y.label) for x in child_of for y in child_of[x]]


child_negative_examples =  [ltn.Clause([ltn.Literal(False, Are_child, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_child_of_"+y.label)
                             for group in [first_group, second_group]
                             for x in group for y in group
                             if y not in child_of[x] and x not in child_of[y]]


parent_of = {barack:{malia,sasha},
             michelle:{sasha,malia},
             sasha:set(),
             malia:set(),
             bill:set(),
             hillary:set(),
             chelsea:set()}


parent_positive_examples = [ltn.Clause([ltn.Literal(True, Are_parent, ltn.Domain_concat([x, y]))],
                                       label = x.label+"_is_parent_of_"+y.label) for x in parent_of for y in parent_of[x]]


parent_negative_examples =  [ltn.Clause([ltn.Literal(False, Are_parent, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_parent_of_"+y.label)
                             for group in [first_group]
                             for x in group for y in group
                             if y not in parent_of[x] and x not in parent_of[y]]

# smoking_positive_examples = [ltn.Clause([ltn.Literal(True,Smokes,x)],
#                                         label=x.label+"_smokes")
#                              for x in [anna,edward,frank,gary,ivan,nick]]
#
# smoking_negative_examples = [ltn.Clause([ltn.Literal(False,Smokes,x)],
#                                         label=x.label+"_does_not_smoke")
#                              for x in everybody if x not in [anna,edward,frank,gary,ivan,nick]]
#
# cancer_positive_examples = [ltn.Clause([ltn.Literal(True,Has_cancer,x)],
#                                        label= x.label+"_has_a_cancer")
#                              for x in [anna,edward]]
#
# cancer_negative_examples = [ltn.Clause([ltn.Literal(False,Has_cancer,x)],
#                                        label = x.label+"_does_not_have_a_cancer")
#                              for x in first_group if x not in [anna,edward]]


save_path = "SFC with bg knowledge/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# KB = ltn.KnowledgeBase("SFC with bg knowledge",
#                        friends_positive_examples +
#                        friends_negative_examples +
#                        smoking_positive_examples +
#                        smoking_negative_examples +
#                        cancer_positive_examples +
#                        cancer_negative_examples +
#                        friendship_is_symmetric +
#                        smokes_implies_has_cancer +
#                        friends_xy_imp_smokes_x_imp_smokes_y +
#                        everybody_has_a_friend,
#                        save_path=save_path)

KB = ltn.KnowledgeBase("SFC with bg knowledge",
                       spouse_is_symmetric+
                       child_p1p2_implies_parent_p2p1+
                       parent_positive_examples +
                       parent_negative_examples +
                       spouse_positive_examples +
                       spouse_negative_examples +
                       child_positive_examples +
                       child_negative_examples,
                       save_path=save_path)

for cl in KB.clauses:
    print(cl.label)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# KB.restore(sess)
val = -sess.run(KB.loss)
print(0,"------>",val, sess.run(KB.tensor))
kb_contains_a_nan = False
# for i in range(100000):

for i in range(5000):
    KB.train(sess)
    sat_level = sess.run(KB.tensor)
    loss_value = sess.run(KB.loss)
    print(i+1," ------> ", sat_level, loss_value)
    if sat_level > .98:
        break

print("saving model")
KB.save(sess)

writepath = save_path+"results_spc_r2.csv"
print("writing results in", writepath)

with open(writepath, "w") as f:
    # result_smoking = sess.run(ltn.Literal(True, Smokes,person).tensor)[:, 0]
    # result_cancer = sess.run(ltn.Literal(True, Has_cancer,person).tensor)[:, 0]
    result_spouse = sess.run(ltn.Literal(True, Are_spouse, p1p2).tensor)[:, 0]
    result_child = sess.run(ltn.Literal(True, Are_child, p1p2).tensor)[:, 0]
    result_parent = sess.run(ltn.Literal(True, Are_parent, p1p2).tensor)[:, 0]
    features = sess.run(person.tensor)
    resultWriter = csv.writer(f, delimiter=";")
    n = len(everybody)
    for i in range(n):
        resultWriter.writerow([everybody[i].label+"'s features are"]+list(features[i]))
        # resultWriter.writerow([everybody[i].label+" smokes", result_smoking[i]])
        # resultWriter.writerow([everybody[i].label+" has cancer", result_cancer[i]])
        for j in range(n):
            resultWriter.writerow([everybody[i].label+" and "+everybody[j].label+" are spouse", result_spouse[j*n+i]])
            resultWriter.writerow([everybody[i].label+" is parent of "+everybody[j].label, result_parent[j*n+i]])
            resultWriter.writerow([everybody[i].label+" is child of "+everybody[j].label, result_child[j*n+i]])
    for cl in KB.clauses:
        resultWriter.writerow([cl.label+" truth value is ", sess.run(cl.tensor)[0][0]])
f.close()
sess.close()










