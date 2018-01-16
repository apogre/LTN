#!/usr/bin/env python
__author__ = "Luciano Serafini"
__copyright__ = "Copyright 2017, Fondazione Bruno Kessler"
__email__ = "serafini@fbk.eu"

import tensorflow as tf
import os
import csv
from persons_all_spc_sample import *
import logictensornetworks as ltn
import sys


ltn.default_layers = 4
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "luk"
ltn.default_aggregator ="hmean"
ltn.default_optimizer = "rmsprop"
ltn.default_clauses_aggregator = "hmean"
ltn.default_fact_penality = 1e-6

everybody = first_group + second_group

print len(first_group), len(second_group), len(everybody)


person = ltn.Domain_union(everybody)

p1p1 = ltn.Domain_concat([person, person])

p1p2 = ltn.Domain_product(person, person)

p1 = ltn.Domain_slice(p1p2, 0, number_of_features)

p2 = ltn.Domain_slice(p1p2, number_of_features, number_of_features*2)

p2p1 = ltn.Domain_concat([p2, p1])

Is_parent = ltn.Predicate("Is_parent", p1p1)
Is_child = ltn.Predicate("Is_child", p1p1)
Is_spouse = ltn.Predicate("Is_spouse", p1p1)

spouse_is_symmetric = \
    [ltn.Clause([ltn.Literal(False, Is_spouse, p1p2),
                 ltn.Literal(True, Is_spouse, p2p1)],
                label="spouse_p1p2_implies_spouse_p2p1")]

child_p1p2_implies_parent_p2p1 = \
    [ltn.Clause([ltn.Literal(False, Is_child, p1p2),
                 ltn.Literal(True, Is_parent, p2p1)],
                label="child_p1p2_implies_parent_p2p1")]


everybody_label = [body.label for body in everybody]

def create_dict(fname, predicate, everybody_label):
    predicate_of = {}
    predicate_key_label = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            persons = [per.rstrip() for per in persons]
            if persons[1] == predicate:
                if persons[0] in everybody_label and persons[2] in everybody_label:
                    predicate_key_label = [pred.label for pred in predicate_of.keys()]
                    if persons[2] not in predicate_key_label:
                        predicate_of[everybody[everybody_label.index(persons[2])]] = [everybody[everybody_label.index(persons[0])]]
                    else:
                        predicate_of[everybody[everybody_label.index(persons[2])]].append(everybody[everybody_label.index(persons[0])])
    return predicate_of, predicate_key_label


parents_of, parents_key_label = create_dict('train_freebase.txt', 'parents', everybody_label)
# print len(parents_of.keys())
# print parents_of.keys()[0].label
# print first_group[0].label


parents_positive_examples = [ltn.Clause([ltn.Literal(True, Is_parent, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_parent_of_"+y.label) for x in parents_of for y in parents_of[x]]

# print profession_of
#
# for k,v in parents_of.iteritems():
#     print k.label
#     for val in v:
#         print val.label

# for ex in parents_positive_examples:
#     print ex.label

# for fex in profession_negative_examples:
#     print fex.label
#
# for group in [first_group]:
#     for x in group:
#         for y in group:
#             print x.label, y.label
#             print parents_of[x]
#             print parents_of[y]


parents_negative_examples = [ltn.Clause([ltn.Literal(False, Is_parent, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_parents_of_"+y.label)
                             for group in [first_group]
                             for x in group for y in group
                             if y not in parents_of.get(x, []) and x not in parents_of.get(y, [])]


children_of, children_key_label = create_dict('spc_freebase_sample.txt', 'children', everybody_label)


children_positive_examples = [ltn.Clause([ltn.Literal(True, Is_child, ltn.Domain_concat([x, y]))],
                                         label=x.label+"_is_children_of_"+y.label) for x in children_of for y in children_of[x]]


children_negative_examples = [ltn.Clause([ltn.Literal(False, Is_child, ltn.Domain_concat([x, y]))],
                                         label=x.label+"_is_not_children_of_"+y.label)
                              for group in [first_group, second_group]
                              for x in group for y in group
                              if y not in children_of.get(x,[]) and x not in children_of.get(y,[])]


spouse_of, spouse_key_label = create_dict('data/train.txt', 'spouse', everybody_label)


spouse_positive_examples = [ltn.Clause([ltn.Literal(True, Is_spouse, ltn.Domain_concat([x, y]))],
                                       label=x.label+"_is_spouse_of_"+y.label) for x in spouse_of for y in spouse_of[x]]


spouse_negative_examples = [ltn.Clause([ltn.Literal(False, Is_spouse, ltn.Domain_concat([x, y]))],
                                        label=x.label+"_is_not_spouse_of_"+y.label)
                            for group in [first_group, second_group]
                            for x in group for y in group
                            if y not in spouse_of.get(x,[]) and x not in spouse_of.get(y,[])]

save_path = "SFC with bg knowledge/"
if not os.path.exists(save_path):
    os.makedirs(save_path)


KB = ltn.KnowledgeBase("SFC with bg knowledge",
                       spouse_is_symmetric+
                       # child_p1p2_implies_parent_p2p1+
                       parents_positive_examples + parents_negative_examples
                       + children_negative_examples + children_positive_examples +
                       spouse_negative_examples + spouse_positive_examples,
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

writepath = save_path+"results_spc_sample_r1_r2.csv"
print("writing results in", writepath)


with open(writepath, "w") as f:
    result_spouse = sess.run(ltn.Literal(True, Is_spouse, p1p2).tensor)[:, 0]
    result_child = sess.run(ltn.Literal(True, Is_child, p1p2).tensor)[:, 0]
    result_parent = sess.run(ltn.Literal(True, Is_parent, p1p2).tensor)[:, 0]

    features = sess.run(person.tensor)
    resultWriter = csv.writer(f, delimiter=";")
    n = len(everybody)
    for i in range(n):
        for j in range(n):
            resultWriter.writerow([everybody[i].label+" and "+everybody[j].label+" are spouse", result_spouse[j*n+i]])
            resultWriter.writerow([everybody[i].label+" is parent of "+everybody[j].label, result_parent[j*n+i]])
            resultWriter.writerow([everybody[i].label+" is child of "+everybody[j].label, result_child[j*n+i]])
    # for cl in KB.clauses:
    #     resultWriter.writerow([cl.label+" truth value is ", sess.run(cl.tensor)[0][0]])
f.close()
sess.close()










