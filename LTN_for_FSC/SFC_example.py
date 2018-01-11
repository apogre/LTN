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

#creates list of Constant objects with properties:
#-value = features_for_generic_peoples 30*30
#tensor- tf.constant(value, dtype=tf.float32)
#parameters-[]
#columns - len(value) = 30
#label - p_i
# const_list = []
# for i in range(number_of_features):
#   const_list.append(ltn.Constant("p"+str(i),features_for_generic_peoples[i]))

# person = ltn.Domain_union(const_list)
#Use constant list and create object with following prop:
#columns = first column = 30
#label = 'a_0_1_2_3_4_5_6_7_8_9'
#parameters = list of individual params, []
#domain = domains-list of object constants
#tensor = list of all individual tensors


#Group 1
anna = ltn.Constant("Anna",    domain=person)
#creating Constant object with following prop:
#label = "Anna"
#columns = 30 - comes from person object
#tensor = random values of size 1*30 defined as variable named "Anna"
#parameters = [tensor]
bob = ltn.Constant("Bob",      domain=person)
chris = ltn.Constant("Chris",  domain=person)
daniel = ltn.Constant("Daniel",domain=person)
edward = ltn.Constant("Edward",domain=person)
frank = ltn.Constant("Frank",  domain=person)
gary = ltn.Constant("Gary",    domain=person)
helen = ltn.Constant("Helen",  domain=person)

#Group 2
mich = ltn.Constant("Mich", domain=person)
kate = ltn.Constant("Kate", domain=person)
ivan = ltn.Constant("Ivan", domain=person)
john = ltn.Constant("John", domain=person)
lars = ltn.Constant("Lars", domain=person)
nick = ltn.Constant("Nick", domain=person)

#cancer knowledge is available
first_group = [anna,bob,chris,daniel,edward,frank,gary,helen]

#cancer knowledge is not available
second_group = [ivan,john,kate,lars,mich,nick]

everybody = first_group + second_group

person = ltn.Domain_union(everybody)
#Use Constant list and create Domain_union object with following prop:
#columns = first column = 30
#label = 'union_of_Anna_Bob_2_3_4_5_6_7_8_9'
#parameters = list of individual params, [[tensor_anna],[tensor_bob]..]
#domain = domains-list of object constants
#tensor = concat of list of all individual tensors - 14 * 30

p1p1 = ltn.Domain_concat([person, person])
#Use Constant lists and create Domain_concat object with following prop:
#columns = 30+30 = 60
#label = 'concat_of_Anna_Bob_2_3_4_5_6_7_8_9'
#parameters = list of individual params, [[tensor_anna],[tensor_bob]..]
#domain = domains-list of object constants
#tensor = concat of list of all individual tensors- column wise (14*60)

p1p2 = ltn.Domain_product(person, person)
#Use Constant lists and create Domain_product object with following prop:
#columns = 30+30 = 60
#label = 'cross product_of_Anna_Bob_2_3_4_5_6_7_8_9'
#parameters = list of individual params, [[tensor_anna],[tensor_bob]..]
#domain = domains-list of object constants
#tensor = concat of cross prodcut of list of all individual tensors- column wise(14*60)


p1 = ltn.Domain_slice(p1p2, 0, number_of_features)
#Use Domain_product object to create Domain_slice object with prop:
#columns = 30 - 0 = 30
#label - "projection_of_cross_product_of.._from_col_0_to_30"
#first part of p1p2

p2 = ltn.Domain_slice(p1p2, number_of_features, number_of_features*2)
#second part of p1p2

p2p1 = ltn.Domain_concat([p2, p1])
#reversing p1p2, each of individual persons

some_friend = ltn.Function("some_friend", person, person)
#create Function object some_friend with prop:
# label = "some_friend"
# in_columns = 30
# columns = 30
# family = "linear"
# domain = person
# M = 31 * 31 random values
# N = 31 * 30 random values
# parameters = N
# --tensor(person)
# --extended_domain = 14*1(ones), 14*30(domain tensors) => 14 * 31
# --result = matmul(extended_domain,N) 14*31


some_friend_of_person = ltn.Term(some_friend, person)
#create Term object some_friend_of_person
#label - some_fried_of_union_of_Anna_Bob_2_3_4_5_6_7_8_9
#parameters - N(31 * 30) + list of individual params(14 * 30)
#domain = person
#function = some_friend
#columns = 30
#tensor = some_friend.tensor(person)


Are_friends = ltn.Predicate("Are_friends", ltn.Domain_concat([person, person]))
# Are_friends = ltn.Predicate("Are_friends", p1p1)?
#label = Are_friends
#domain = domain
#defined = None
#number of layer = 5
#W = 5 * 61 * 61 -random numbers
#u = 5 * 1 - ones
#tensor
#--X =  14 * (60+1)
#--XW =    matmul(X(5 * 14 * 61)?, W(5*61*61)) 

# Smokes(person)

Smokes = ltn.Predicate("Smokes", person)
#label = Smokes
#domain = domain
#defined = None
#number of layer = 5
#W = 5 * 31 * 31 -random numbers
#u = 5 * 1 - ones
#tensor
#--X =  14 * (30+1)
#--XW =    matmul(X(5 * 14 * 31)?, W(5*31*31))
#returns [0,1]

Has_cancer = ltn.Predicate("has_cancer", person)
#parameters- [W, u]
#returns [0,1]


smokes_implies_has_cancer = \
    [ltn.Clause([ltn.Literal(False, Smokes, person),
                 ltn.Literal(True, Has_cancer, person)], label="smoking_implies_cancer")]

#ltn.Literal(True, Has_cancer, person)
#label - none
# Predicate - Has_cancer
# polarity - True
# domain - person
# parameters - [W,u]+list of all object tensors
# if polarity:
#   tensor - Has_cancer.tensor(person)->[0,1]
# else:
#   tensor - 1-Smokes.tensor(person)->[0,1]

#ltn.Clause([lit1,lit2],label="abc")
#Clause object is created with props:
# weight = 1
# label = "smoking_implies_cancer"
# litearals = [lit1, lit2]
# tensor => [0,1]
# -disjunction_of_literals(literals,label="no_label",tnorm=product,aggregator=min):
# --result = 1.0-tf.reduce_prod(1.0-literals_tensor,1,keep_dims=True)
# --return tf.reduce_min(result, keep_dims=True,name=label)
# --min(1.0, return)

#predicate - [Smokes, Has_Cancer]
#parameters - [lit1.parameters, lit2.parameters]

friends_xy_imp_smokes_x_imp_smokes_y = \
    [ltn.Clause([ltn.Literal(False, Are_friends, p1p2),
                 ltn.Literal(False, Smokes, p1),
                 ltn.Literal(True, Smokes, p2)],
                label="friends_p1p2_and_smokes_p1_implies_smokes_p2")]

friendship_is_symmetric = \
    [ltn.Clause([ltn.Literal(False, Are_friends, p1p2),
                 ltn.Literal(True, Are_friends, p2p1)],
                label="friends_p1p2_implies_friends_p2p1")]

everybody_has_a_friend = [ltn.Clause([ltn.Literal(True, Are_friends, ltn.Domain_concat([person, some_friend_of_person]))],
                                     label="everybody_has_a_friend")]


friends_of = {anna:{bob,edward,frank,gary},
              bob:{chris},
              chris:{daniel},
              daniel:set(),
              edward:{frank},
              frank:set(),
              gary:{helen},
              helen:set(),
              ivan:{john,mich},
              john:set(),
              kate:{lars},
              lars:set(),
              mich:{nick},
              nick:set()}


friends_positive_examples = [ltn.Clause([ltn.Literal(True,Are_friends,ltn.Domain_concat([x,y]))],
                label = x.label+"_and_"+y.label+"_are_friends") for x in friends_of for y in friends_of[x]]


friends_negative_examples =  [ltn.Clause([ltn.Literal(False,Are_friends,ltn.Domain_concat([x,y]))],
                label = x.label+"_and_"+y.label+"_are_not_friends")
                              for group in [first_group,second_group]
                              for x in group for y in group
                              if y not in friends_of[x] and x not in friends_of[y]]

smoking_positive_examples = [ltn.Clause([ltn.Literal(True,Smokes,x)],
                                        label=x.label+"_smokes")
                             for x in [anna,edward,frank,gary,ivan,nick]]

smoking_negative_examples = [ltn.Clause([ltn.Literal(False,Smokes,x)],
                                        label=x.label+"_does_not_smoke")
                             for x in everybody if x not in [anna,edward,frank,gary,ivan,nick]]

cancer_positive_examples = [ltn.Clause([ltn.Literal(True,Has_cancer,x)],
                                       label= x.label+"_has_a_cancer")
                             for x in [anna,edward]]

cancer_negative_examples = [ltn.Clause([ltn.Literal(False,Has_cancer,x)],
                                       label = x.label+"_does_not_have_a_cancer")
                             for x in first_group if x not in [anna,edward]]


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
                       friends_positive_examples +
                       friends_negative_examples +
                       smoking_positive_examples +
                       smoking_negative_examples +
                       cancer_positive_examples +
                       cancer_negative_examples,
                       save_path=save_path)

for cl in KB.clauses:
    print(cl.label)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# KB.restore(sess)
val = -sess.run(KB.loss)
print(0,"------>",val,sess.run(KB.tensor))
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

writepath = save_path+"results_1.csv"
print("writing results in", writepath)
with open(writepath,"w") as f:
    result_smoking = sess.run(ltn.Literal(True,Smokes,person).tensor)[:,0]
    result_cancer = sess.run(ltn.Literal(True,Has_cancer,person).tensor)[:,0]
    result_friends = sess.run(ltn.Literal(True,Are_friends,p1p2).tensor)[:,0]
    features = sess.run(person.tensor)
    resultWriter = csv.writer(f, delimiter =";")
    n = len(everybody)
    for i in range(n):
        resultWriter.writerow([everybody[i].label+"'s features are"]+list(features[i]))
        resultWriter.writerow([everybody[i].label+" smokes",result_smoking[i]])
        resultWriter.writerow([everybody[i].label+" has cancer",result_cancer[i]])
        for j in range(n):
            resultWriter.writerow([everybody[i].label+" and "+everybody[j].label+" are friends",result_friends[j*n+i]])
    for cl in KB.clauses:
        resultWriter.writerow([cl.label+" truth value is ",sess.run(cl.tensor)[0][0]])
f.close()
sess.close()










