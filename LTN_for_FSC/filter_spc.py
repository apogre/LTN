import sys
predicate = ['children', 'spouse', 'parents']


def write_filtered_data(predicate):
    with open('data/train.txt') as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            if persons[1] in predicate:
                with open('data/spc_freebase.txt', 'a') as f:
                    f.write(line)


# write_filtered_data(predicate)
# sys.exit(0)

def get_unique_persons(fname):
    all_persons = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            persons.pop(1)
            persons = [per.rstrip() for per in persons]
            all_persons.extend(persons)
    return all_persons

all_persons_init = get_unique_persons('data/spc_freebase.txt')

all_persons_unique = list(set(all_persons_init))

# print all_persons_unique
print len(all_persons_unique)

family = {}
for u_person in all_persons_unique:
    with open('data/spc_freebase.txt') as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            predicate = persons[1]
            persons.pop(1)
            persons = [per.rstrip() for per in persons]
            if u_person in persons:
                if u_person not in family.keys():
                    family[u_person] = {'relation':[predicate],'entities':[u_person], 'triples':[line]}
                else:
                    family[u_person]['triples'].append(line)
                    if predicate not in family[u_person]['relation']:
                        family[u_person]['relation'].append(predicate)
                if u_person == persons[0]:
                    if persons[1] in all_persons_unique:
                        family[u_person]['entities'].append(persons[1])
                        all_persons_unique.remove(persons[1])
                else:
                    if persons[0] in all_persons_unique:
                        family[u_person]['entities'].append(persons[0])
                        all_persons_unique.remove(persons[0])

# print family
print len(family.keys())
filtered_family = {}
filtered_family_train = {}
train_tuples = []
filtered_family_test = {}
test_tuples = []
count = 0
for k,v in family.iteritems():
    if len(v.get('relation')) == 3:
        count +=1
        if count < 5:
            filtered_family_train[k] = v
            train_tuples.extend(v.get('triples'))
        elif count>10:
            break
        else:
            filtered_family_test[k] = v
            test_tuples.extend(v.get('triples'))

print count
print len(filtered_family)
# print filtered_family
with open('train_freebase_sample.txt', 'a') as f:
    for tup in train_tuples:
        f.write(tup)

with open('test_freebase_all_sample.txt', 'a') as f:
    for tup in test_tuples:
        f.write(tup)

with open('test_freebase_sample.txt', 'a') as f:
    with open('test_freebase_parents_sample.txt', 'a') as g:
        for tup in test_tuples:
            if 'parents' not in tup:
                f.write(tup)
            else:
                g.write(tup)

