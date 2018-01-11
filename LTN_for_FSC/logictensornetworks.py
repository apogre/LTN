#!/usr/bin/env python
__author__ = "Luciano Serafini"
__copyright__ = "Copyright 2017, Fondazione Bruno Kessler"
__email__ = "serafini@fbk.eu"

import sys
import tensorflow as tf
import numpy as np
import pdb

default_layers = 5
default_smooth_factor = 0.0000001
default_tnorm = "product"
default_optimizer = "gd"
default_aggregator = "min"
default_clauses_aggregator = "min"
default_fact_penality = 0.0

BIAS = 0.0


def train_op(loss, optimization_algorithm):
    if optimization_algorithm == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.01,learning_rate_power=-0.5)
    if optimization_algorithm == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    if optimization_algorithm == "ada":
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    if optimization_algorithm == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005,decay=0.9)
    return optimizer.minimize(loss)


def PR(tensor):
    np.set_printoptions(threshold=np.nan)
    result = tf.Print(tensor, [tf.shape(tensor), tensor.name, tensor],summarize=20)
    return result


def FIN(tensor):
    np.set_printoptions(threshold=np.nan)
    result = tf.verify_tensor_all_finite(tensor, "is NaN " + tensor.name)
    return result


def disjunction_of_literals(literals,label="no_label", tnorm=default_tnorm, aggregator=default_aggregator):
    list_of_literal_tensors = [lit.tensor for lit in literals]
    literals_tensor = tf.concat(list_of_literal_tensors, 1, name=label)

    if tnorm == "product":
        result = 1.0-tf.reduce_prod(1.0-literals_tensor, 1, keep_dims=True)
    elif tnorm == "yager2":
        result = tf.minimum(1.0, tf.sqrt(tf.reduce_sum(tf.square(literals_tensor),1, keep_dims=True)))

    elif tnorm == "luk":
        result = tf.minimum(1.0, tf.reduce_sum(literals_tensor, 1, keep_dims=True))

    elif tnorm == "goedel":
        result = tf.reduce_max(literals_tensor,1,keep_dims=True,name=label)

    if aggregator == "product":
        return tf.reduce_prod(result,keep_dims=True,name=label)
    elif aggregator == "mean":
        return tf.reduce_mean(result,keep_dims=True,name=label)

    elif aggregator == "gmean":
        return tf.exp(tf.multiply(tf.reduce_sum(tf.log(result), keep_dims=True),
                      tf.reciprocal(tf.to_float(tf.size(result)))), name=label)

    elif aggregator == "hmean":
        return tf.div(tf.to_float(tf.size(result)), tf.reduce_sum(tf.reciprocal(result), keep_dims=True), name=label)

    elif aggregator == "min":
        return tf.reduce_min(result, keep_dims=True, name=label)


def smooth(parameters, smooth_factor=default_smooth_factor):
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat([tf.expand_dims(tf.reduce_sum(tf.square(par)), 0) for par in parameters], 0), 1))
    return tf.multiply(smooth_factor, norm_of_omega)


class Domain:
    def __init__(self,columns,dom_type="float", label=None):
        self.dom_type = dom_type
        self.columns = columns
        if label is None:
            label = "domain"
        self.label = label
        self.parameters = []
        self.tensor = tf.placeholder(self.dom_type, shape=[None, self.columns], name=self.label)


class Domain_concat(Domain):
    def __init__(self, domains, label = None):
        self.columns = sum([dom.columns for dom in domains])
        if label is None:
            self.label = "concatenation_of_" + "_".join([dom.label for dom in domains])
        else:
            self.label = label
        self.parameters = [par for dom in domains for par in dom.parameters]
        self.domains = domains
        self.tensor = tf.concat([dom.tensor for dom in self.domains], 1)


class Domain_union(Domain):
    def __init__(self, domains, label = None):
        self.columns = domains[0].columns
        if label is None:
            self.label = "union_of_" + "_".join([dom.label for dom in domains])
        else:
        	self.label = label
        self.parameters = [par for dom in domains for par in dom.parameters]
        self.domains = domains
        self.tensor = tf.concat([dom.tensor for dom in self.domains],0)
        #linear concatenation by row


class Domain_product(Domain):
    def __init__(self, dom1, dom2, label=None):
        self.columns = dom1.columns + dom2.columns
        if label is None:
            self.label = "cross_product_of_" + "_".join([dom.label for dom in [dom1,dom2]])
        else:
        	self.label = label
        self.parameters = [par for dom in [dom1, dom2] for par in dom.parameters]
        tensor1 = tf.tile(dom1.tensor, (dom2.tensor.shape[0], 1))
        tensor2 = tf.reshape(tf.tile(dom2.tensor, (1, dom1.tensor.shape[0])),
                             (dom1.tensor.shape[0]*dom2.tensor.shape[0], dom2.tensor.shape[1]))
        self.tensor = tf.concat([tensor1, tensor2], 1)


class Domain_slice(Domain):
    def __init__(self, domain, begin_column, end_column):
        self.columns = end_column - begin_column
        self.label = "projection_of_" + domain.label + "_from_column_"+str(begin_column) + "_to_column_" + str(end_column)
        self.parameters = domain.parameters
        self.domain = domain
        self.begin_column = begin_column
        self.end_column = end_column
        self.tensor = tf.slice(domain.tensor, [0, begin_column], [domain.tensor.shape[0], end_column - begin_column])


class Constant(Domain):
    def __init__(self, label, value=None, domain=None):
        self.label = label
        if value is not None:
            self.tensor = tf.constant(value, dtype=tf.float32)
            self.parameters = []
            self.columns = len(value)
        else:
            self.columns = domain.columns
            self.tensor = tf.Variable(tf.random_normal([1, domain.columns], mean=0,stddev=1),name=self.label)
            self.parameters = [self.tensor]


class Function:
    def __init__(self, label, domain, range, value=None,mean=0.0,stddev=0.5,family="linear"):
        self.label = label
        self.in_columns = domain.columns
        self.columns = range.columns
        self.domain = domain
        self.family = family
        if value is None:
            self.M = tf.Variable(tf.random_normal([self.in_columns+1,
                                                   self.in_columns+1],mean=mean,stddev=stddev),
                                 name="M_"+self.label)
            self.N = tf.Variable(tf.random_normal([self.in_columns+1,
                                                   self.columns],mean=mean,stddev=stddev),
                                 name="N_"+self.label)
            if family == "linear":
                self.parameters = [self.N]
            if family == "fnn":
                self.parameters = [self.M,self.N]
            self.is_defined = False
        else:
            self.parameters = []
            self.is_defined = True
            self.value = value

    def tensor(self, domain=None):
        if domain is None:
            domain = self.domain
        if self.is_defined:
            result = apply(self.value, [domain])
        else:
            extended_domain = tf.concat(
                [
                    tf.ones(
                        (tf.shape(
                            domain.tensor
                        )[0], 1)
                    ), domain.tensor
                ], 1)
            if self.family == "fnn":
                h = tf.nn.sigmoid(tf.matmul(extended_domain, self.M))
                result = tf.matmul(h, self.N)
            if self.family == "linear":
                result = tf.matmul(extended_domain, self.N, name=self.label)
        return FIN(result)


class Term(Domain):
    def __init__(self,function, domain):
        self.label = function.label+"_of_"+domain.label
        self.parameters = function.parameters + domain.parameters
        self.domain = domain
        self.function = function
        self.columns = function.columns
        self.tensor = self.function.tensor(self.domain)


class Predicate:
    def __init__(self, label, domain, layers=default_layers, defined=None, **kwargs):
        self.label = label
        if not isinstance(domain, Domain):
            domain = Domain_concat(domain)
        self.domain = domain
        self.defined = defined
        self.number_of_layers = layers
        if self.defined is None:
            self.W = tf.matrix_band_part(tf.Variable(tf.random_normal([layers, self.domain.columns + 1, \
                                                                       self.domain.columns + 1], stddev=0.0)), 0, -1, \
                                         name="W" + label)  # upper triangualr matrix
            self.u = tf.Variable(tf.ones([layers, 1]), name="u" + label)
            self.parameters = [self.W, self.u]
        else:
            self.parameters = []

    def tensor(self, domain=None):
        if domain is None:
            domain = self.domain
        if self.defined is not None:
            return self.defined(domain.tensor)
        else:
            X = tf.concat([tf.ones((tf.shape(domain.tensor)[0], 1)), domain.tensor], 1)
            XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.number_of_layers, 1, 1]), self.W)
            XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), squeeze_dims=[1])
            gX = tf.matmul(tf.tanh(XWX), self.u)
            result = tf.sigmoid(gX, name=self.label + "_at_" + domain.label)
            return result


class In_range(Predicate):
    def __init__(self,domain,lower,upper,label="inrange",sharpness=10.0):
        self.label = label
        self.domain = domain
        self.parameters = []
        self.lower = tf.constant(lower,dtype=tf.float32)
        self.upper = tf.constant(upper,dtype=tf.float32)
        self.normalize = tf.square(tf.divide(tf.subtract(self.upper,
                                                         self.lower),2.0))
        self.sharpness = sharpness

    def tensor(self,domain=None):
        if domain is None:
            domain = self.domain
        dom_leq_upper = tf.exp(tf.minimum(0.0,tf.reduce_min(tf.subtract(self.upper,domain.tensor),1,keep_dims=True))/self.sharpness)
        dom_geq_lower = tf.exp(tf.minimum(0.0,tf.reduce_min(tf.subtract(domain.tensor,self.lower),1,keep_dims=True))/self.sharpness)
        return dom_geq_lower*dom_leq_upper



class Less_than(Predicate):
    def __init__(self,domain1,domain2,label,sharpness=10.0):
        self.label = label
        self.domain = Domain_concat([domain1,domain2])
        self.parameters = []
        self.sharpness = sharpness

    def tensor(self,domain=None):
        if domain is None:
            domain = self.domain
        return tf.reduce_min(
            tf.nn.sigmoid(tf.multiply(self.sharpness,
                                     tf.subtract(domain.tensor[:,domain.tensor.shape[1]/2:],
                                                 domain.tensor[:,:domain.tensor.shape[1]/2]))),keep_dims=True)

class Equal(Predicate):
    def __init__(self, label, domain,diameter=1.0):
        self.label = label
        self.domain = domain
        self.parameters = []
        self.diameter = diameter

    def tensor(self,dom=None):
        if dom is None:
            dom = self.domain
        dom1_tensor = dom.tensor[:,:dom.columns/2]
        dom2_tensor = dom.tensor[:,dom.columns/2:]
        delta = tf.sqrt(tf.reduce_sum(tf.square(dom1_tensor - dom2_tensor)  ,1,keep_dims=True))
        return tf.exp(-tf.divide(delta,self.diameter))

class Literal:
    def __init__(self,polarity,predicate,domain=None,label=None):
        global BIAS
        self.label=label
        self.predicate = predicate
        self.polarity = polarity
        if domain is None:
            self.domain = predicate.domain
            self.parameters = predicate.parameters
        else:
            if not isinstance(domain, Domain):
                domain = Domain_concat(domain)
            self.domain = domain
            self.parameters = predicate.parameters + domain.parameters

        if polarity:
            self.tensor = predicate.tensor(domain)
            if default_fact_penality > 0.0:
                BIAS += default_fact_penality*tf.reduce_sum(self.tensor)
        else:
            self.tensor = 1-predicate.tensor(domain)
            if default_fact_penality < 0.0:
                BIAS += default_fact_penality*tf.reduce_sum(self.tensor)


class Clause:
    def __init__(self, literals, label="Clause", weight=1.0, aggregator=default_aggregator, tnorm=default_tnorm, **kwargs):
        self.weight = weight
        self.label = label
        self.literals = literals
        self.tensor = tf.minimum(1.0, tf.divide(disjunction_of_literals(self.literals, label=label,\
                                                                        aggregator=aggregator, tnorm=tnorm), self.weight))
        self.predicates = set([lit.predicate for lit in self.literals])
        self.parameters = [par for lit in literals for par in lit.parameters]


class KnowledgeBase:
    def __init__(self, label, clauses,save_path=""):
        clauses_aggregator = default_clauses_aggregator
        smooth_factor = default_smooth_factor
        optimizer = default_optimizer
        self.label = label.replace(' ','_')
        self.clauses = clauses
        self.parameters = [par for cl in self.clauses for par in cl.parameters]
        if not self.clauses:
            self.tensor = tf.constant(1.0)
        else:
            clauses_value_tensor = tf.concat([cl.tensor for cl in clauses],0)
            if clauses_aggregator == "min":
                self.tensor = tf.reduce_min(clauses_value_tensor,name=self.label)
            if clauses_aggregator == "mean":
                self.tensor = tf.reduce_mean(clauses_value_tensor,name=self.label)
            if clauses_aggregator == "hmean":
                self.tensor = tf.div(tf.to_float(tf.size(clauses_value_tensor)),
                                        tf.reduce_sum(tf.reciprocal(clauses_value_tensor)),name=self.label)
            if clauses_aggregator == "wmean":
                weights_tensor = tf.constant([cl.weight for cl in clauses])
                self.tensor = tf.div(tf.reduce_sum(tf.multiply(weights_tensor, clauses_value_tensor)),tf.reduce_sum(weights_tensor),name=self.label)

            self.loss = tf.subtract(BIAS+smooth(self.parameters, smooth_factor=smooth_factor),self.tensor,name="Loss")
        self.save_path = save_path
        self.train_op = train_op(self.loss,optimizer)
        self.saver = tf.train.Saver()

    def save(self,sess,version=""):
        save_path = self.saver.save(sess,self.save_path+self.label+version+".ckpt")

    def restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self,sess,feed_dict={}):
        return sess.run(self.train_op,feed_dict)

    def is_nan(self,sess,feed_dict={}):
        return sess.run(tf.is_nan(self.tensor),feed_dict)
