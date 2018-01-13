#!/usr/bin/env python
__author__ = "Luciano Serafini"
__copyright__ = "Copyright 2017, Fondazione Bruno Kessler"
__email__ = "serafini@fbk.eu"

import tensorflow as tf
import os
import csv
from persons_all import *
import logictensornetworks as ltn
import sys


ltn.default_layers = 4
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "luk"
ltn.default_aggregator ="hmean"
ltn.default_optimizer = "rmsprop"
ltn.default_clauses_aggregator = "hmean"
ltn.default_fact_penality = 1e-6

# first_group = [barack, michelle, sasha, malia]
#
# second_group = [bill, hillary, chelsea]
#
# everybody = first_group + second_group

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
# Are_spouse = ltn.Predicate("Are_spouse", p1p1)
# Are_parent = ltn.Predicate("Are_parent", p1p1)
# Are_child = ltn.Predicate("Are_child", p1p1)
Is_gender = ltn.Predicate("Is_gender", p1p1)
Is_profession = ltn.Predicate("Is_profession", p1p1)
Is_ethnicity = ltn.Predicate("Is_ethnicity", p1p1)
Is_religion = ltn.Predicate("Is_religion", p1p1)
Is_location = ltn.Predicate("Is_location", p1p1)
Is_parents = ltn.Predicate("Is_parents", p1p1)
Is_nationality = ltn.Predicate("Is_nationality", p1p1)
Is_institution = ltn.Predicate("Is_institution", p1p1)
Is_children = ltn.Predicate("Is_children", p1p1)
Is_spouse = ltn.Predicate("Is_spouse", p1p1)
Is_place_of_birth = ltn.Predicate("Is_place_of_birth", p1p1)
Is_cause_of_death = ltn.Predicate("Is_cause_of_death", p1p1)
Is_place_of_death = ltn.Predicate("Is_place_of_death", p1p1)
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
# spouse_is_symmetric = \
#     [ltn.Clause([ltn.Literal(False, Are_spouse, p1p2),
#                  ltn.Literal(True, Are_spouse, p2p1)],
#                 label="spouse_p1p2_implies_spouse_p2p1")]

# child_p1p2_implies_parent_p2p1 = \
#     [ltn.Clause([ltn.Literal(False, Are_child, p1p2),
#                  ltn.Literal(True, Are_parent, p2p1)],
#                 label="child_p1p2_implies_parent_p2p1")]
#
# everybody_has_a_friend = [ltn.Clause([ltn.Literal(True, Are_friends, ltn.Domain_concat([person, some_friend_of_person]))],
#                                      label="everybody_has_a_friend")]

everybody_label = [body.label for body in everybody]


def create_dict(fname, predicate, everybody_label):
    gender_of = {}
    gender_key_label = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            persons = [per.rstrip() for per in persons]
            if persons[1] == predicate:
                if persons[0] in everybody_label and persons[2] in everybody_label:
                    gender_key_label = [spouse.label for spouse in gender_of.keys()]
                    if persons[2] not in gender_key_label:
                        gender_of[everybody[everybody_label.index(persons[2])]] = [everybody[everybody_label.index(persons[0])]]
                    else:
                        gender_of[everybody[everybody_label.index(persons[2])]].append(everybody[everybody_label.index(persons[0])])
    return gender_of, gender_key_label


gender_of, gender_key_label = create_dict('data/train.txt', 'gender', everybody_label)


gender_positive_examples = [ltn.Clause([ltn.Literal(True, Is_gender, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_gender_of_"+y.label) for x in gender_of for y in gender_of[x]]


gender_negative_examples = [ltn.Clause([ltn.Literal(False, Is_gender, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_gender_of_"+y.label)
                             for x in gender_set for y in everybody
                             if y not in gender_of[x]]

profession_of, profession_key_label = create_dict('data/train.txt', 'profession', everybody_label)


profession_positive_examples = [ltn.Clause([ltn.Literal(True, Is_profession, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_profession_of_"+y.label) for x in profession_of for y in profession_of[x]]


profession_negative_examples = [ltn.Clause([ltn.Literal(False, Is_profession, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_profession_of_"+y.label)
                             for x in profession_set for y in everybody
                             if y not in profession_of.get(x,[])]

ethnicity_of, ethnicity_key_label = create_dict('data/train.txt', 'ethnicity', everybody_label)


ethnicity_positive_examples = [ltn.Clause([ltn.Literal(True, Is_ethnicity, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_ethnicity_of_"+y.label) for x in ethnicity_of for y in ethnicity_of[x]]


ethnicity_negative_examples = [ltn.Clause([ltn.Literal(False, Is_ethnicity, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_ethnicity_of_"+y.label)
                             for x in ethnicity_set for y in everybody
                             if y not in ethnicity_of[x]]

place_of_birth_of, place_of_birth_key_label = create_dict('data/train.txt', 'place_of_birth', everybody_label)


place_of_birth_positive_examples = [ltn.Clause([ltn.Literal(True, Is_place_of_birth, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_place_of_birth_of_"+y.label) for x in place_of_birth_of for y in place_of_birth_of[x]]


place_of_birth_negative_examples = [ltn.Clause([ltn.Literal(False, Is_place_of_birth, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_place_of_birth_of_"+y.label)
                             for x in place_of_birth_set for y in everybody
                             if y not in place_of_birth_of.get(x,[])]

religion_of, religion_key_label = create_dict('data/train.txt', 'religion', everybody_label)


religion_positive_examples = [ltn.Clause([ltn.Literal(True, Is_religion, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_religion_of_"+y.label) for x in religion_of for y in religion_of[x]]


religion_negative_examples = [ltn.Clause([ltn.Literal(False, Is_religion, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_religion_of_"+y.label)
                             for x in religion_set for y in everybody
                             if y not in religion_of[x]]


location_of, location_key_label = create_dict('data/train.txt', 'location', everybody_label)


location_positive_examples = [ltn.Clause([ltn.Literal(True, Is_location, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_location_of_"+y.label) for x in location_of for y in location_of[x]]


location_negative_examples = [ltn.Clause([ltn.Literal(False, Is_location, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_location_of_"+y.label)
                             for x in location_set for y in everybody
                             if y not in location_of.get(x,[])]

cause_of_death_of, cause_of_death_key_label = create_dict('data/train.txt', 'cause_of_death', everybody_label)


cause_of_death_positive_examples = [ltn.Clause([ltn.Literal(True, Is_cause_of_death, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_cause_of_death_of_"+y.label) for x in cause_of_death_of for y in cause_of_death_of[x]]


cause_of_death_negative_examples = [ltn.Clause([ltn.Literal(False, Is_cause_of_death, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_cause_of_death_of_"+y.label)
                             for x in cause_of_death_set for y in everybody
                             if y not in cause_of_death_of[x]]


parents_of, parents_key_label = create_dict('data/train.txt', 'parents', everybody_label)


parents_positive_examples = [ltn.Clause([ltn.Literal(True, Is_parents, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_parents_of_"+y.label) for x in parents_of for y in parents_of[x]]


parents_negative_examples = [ltn.Clause([ltn.Literal(False, Is_parents, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_parents_of_"+y.label)
                             for x in parents_set for y in everybody
                             if y not in parents_of.get(x,[])]

institution_of, institution_key_label = create_dict('data/train.txt', 'institution', everybody_label)


institution_positive_examples = [ltn.Clause([ltn.Literal(True, Is_institution, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_institution_of_"+y.label) for x in institution_of for y in institution_of[x]]


institution_negative_examples = [ltn.Clause([ltn.Literal(False, Is_institution, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_institution_of_"+y.label)
                             for x in institution_set for y in everybody
                             if y not in institution_of[x]]

children_of, children_key_label = create_dict('data/train.txt', 'children', everybody_label)


children_positive_examples = [ltn.Clause([ltn.Literal(True, Is_children, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_children_of_"+y.label) for x in children_of for y in children_of[x]]


children_negative_examples = [ltn.Clause([ltn.Literal(False, Is_children, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_children_of_"+y.label)
                             for x in children_set for y in everybody
                             if y not in children_of.get(x,[])]


spouse_of, spouse_key_label = create_dict('data/train.txt', 'spouse', everybody_label)


spouse_positive_examples = [ltn.Clause([ltn.Literal(True, Is_spouse, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_spouse_of_"+y.label) for x in spouse_of for y in spouse_of[x]]


spouse_negative_examples = [ltn.Clause([ltn.Literal(False, Is_spouse, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_spouse_of_"+y.label)
                             for x in spouse_set for y in everybody
                             if y not in spouse_of[x]]

nationality_of, nationality_key_label = create_dict('data/train.txt', 'nationality', everybody_label)


nationality_positive_examples = [ltn.Clause([ltn.Literal(True, Is_nationality, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_nationality_of_"+y.label) for x in nationality_of for y in nationality_of[x]]


nationality_negative_examples = [ltn.Clause([ltn.Literal(False, Is_nationality, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_nationality_of_"+y.label)
                             for x in nationality_set for y in everybody
                             if y not in nationality_of[x]]

place_of_death_of, place_of_death_key_label = create_dict('data/train.txt', 'place_of_death', everybody_label)


place_of_death_positive_examples = [ltn.Clause([ltn.Literal(True, Is_place_of_death, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_place_of_death_of_"+y.label) for x in place_of_death_of for y in place_of_death_of[x]]


place_of_death_negative_examples = [ltn.Clause([ltn.Literal(False, Is_place_of_death, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_place_of_death_of_"+y.label)
                             for x in place_of_death_set for y in everybody
                             if y not in place_of_death_of[x]]


# print profession_of
#
# for k,v in profession_of.iteritems():
#     print k.label
#     for val in v:
#         print val.label
#
# for ex in profession_positive_examples:
#     print ex.label
#
# for fex in profession_negative_examples:
#     print fex.label

# sys.exit()

# spouse_of = {barack:[michelle],
#              michelle:set(),
#              sasha:set(),
#              malia:set(),
#              bill:[hillary],
#              hillary:set(),
#              chelsea:set()}


# spouse_positive_examples = [ltn.Clause([ltn.Literal(True, Are_spouse, ltn.Domain_concat([x, y]))],
#                                        label=x.label+"_and_"+y.label+"_are_spouse") for x in spouse_of for y in spouse_of[x]]
#
#
# spouse_negative_examples = [ltn.Clause([ltn.Literal(False, Are_spouse, ltn.Domain_concat([x, y]))],
#                                         label=x.label+"_and_"+y.label+"_are_not_spouse")
#                              # for group in [first_group, second_group]
#                              for x in everybody for y in everybody
#                              if y not in spouse_of[x] and x not in spouse_of[y]]
#
#
# child_of = {barack:set(),
#              michelle:set(),
#              sasha:[barack,michelle],
#              malia:[michelle,barack],
#              bill:set(),
#              hillary:set(),
#              chelsea:[bill,hillary]}
#
#
# child_positive_examples = [ltn.Clause([ltn.Literal(True, Are_child, ltn.Domain_concat([x, y]))],
#                                        label=x.label+"_is_child_of_"+y.label) for x in child_of for y in child_of[x]]
#
#
# child_negative_examples = [ltn.Clause([ltn.Literal(False, Are_child, ltn.Domain_concat([x, y]))],
#                                         label=x.label+"_is_not_child_of_"+y.label)
#                              # for group in [first_group, second_group]
#                              for x in everybody for y in everybody
#                              if y not in child_of[x] and x not in child_of[y]]
#
#
# parent_of = {barack:[malia,sasha],
#              michelle:[sasha,malia],
#              sasha:set(),
#              malia:set(),
#              bill:set(),
#              hillary:set(),
#              chelsea:set()}
#
#
# parent_positive_examples = [ltn.Clause([ltn.Literal(True, Are_parent, ltn.Domain_concat([x, y]))],
#                                        label=x.label+"_is_parent_of_"+y.label) for x in parent_of for y in parent_of[x]]
#
#
# parent_negative_examples =  [ltn.Clause([ltn.Literal(False, Are_parent, ltn.Domain_concat([x, y]))],
#                                         label=x.label+"_is_not_parent_of_"+y.label)
#                              # for group in [first_group]
#                              for x in everybody for y in everybody
#                              if y not in parent_of[x] and x not in parent_of[y]]

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
                       # spouse_is_symmetric+
                       # child_p1p2_implies_parent_p2p1+
                       gender_positive_examples +
                       gender_negative_examples +
                       profession_negative_examples +
                       ethnicity_negative_examples +
                       ethnicity_positive_examples +
                       religion_negative_examples +
                       religion_positive_examples +
                       cause_of_death_negative_examples +
                       cause_of_death_positive_examples +
                       institution_negative_examples +
                       institution_positive_examples +
                       nationality_negative_examples +
                       nationality_positive_examples +
                       place_of_death_negative_examples +
                       place_of_death_positive_examples +
                       place_of_birth_negative_examples +
                       place_of_birth_positive_examples +
                       location_negative_examples +
                       location_positive_examples +
                       parents_positive_examples + parents_negative_examples 
                       + children_negative_examples + children_positive_examples +
                       spouse_negative_examples + spouse_positive_examples +
                       profession_positive_examples,
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

writepath = save_path+"results_spc_5f.csv"
print("writing results in", writepath)

with open(writepath, "w") as f:
    # result_smoking = sess.run(ltn.Literal(True, Smokes,person).tensor)[:, 0]
    # result_cancer = sess.run(ltn.Literal(True, Has_cancer,person).tensor)[:, 0]
    # result_spouse = sess.run(ltn.Literal(True, Are_spouse, p1p2).tensor)[:, 0]
    # result_child = sess.run(ltn.Literal(True, Are_child, p1p2).tensor)[:, 0]
    # result_parent = sess.run(ltn.Literal(True, Are_parent, p1p2).tensor)[:, 0]
    result_gender = sess.run(ltn.Literal(True, Is_gender, p1p2).tensor)[:, 0]
    result_profession = sess.run(ltn.Literal(True, Is_profession, p1p2).tensor)[:, 0]

    features = sess.run(person.tensor)
    resultWriter = csv.writer(f, delimiter=";")
    n = len(everybody)
    for i in range(n):
        # resultWriter.writerow([everybody[i].label+"'s features are"]+list(features[i]))
        # resultWriter.writerow([everybody[i].label+" smokes", result_smoking[i]])
        # resultWriter.writerow([everybody[i].label+" has cancer", result_cancer[i]])
        for j in range(n):
            # resultWriter.writerow([everybody[i].label+" and "+everybody[j].label+" are spouse", result_spouse[j*n+i]])
            # resultWriter.writerow([everybody[i].label+" is parent of "+everybody[j].label, result_parent[j*n+i]])
            # resultWriter.writerow([everybody[i].label+" is child of "+everybody[j].label, result_child[j*n+i]])
            resultWriter.writerow([everybody[i].label+" is gender of "+everybody[j].label, result_gender[j*n+i]])
            resultWriter.writerow([everybody[i].label+" is profession of "+everybody[j].label, result_profession[j*n+i]])
    # for cl in KB.clauses:
    #     resultWriter.writerow([cl.label+" truth value is ", sess.run(cl.tensor)[0][0]])
f.close()
sess.close()










