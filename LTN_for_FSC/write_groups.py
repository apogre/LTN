import pickle

def form_groups(fname):
    all_persons = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            persons = [per.rstrip() for per in persons]
            persons.pop(1)
            all_persons.extend(persons)
    return all_persons

all_persons_train = []

for predicate in ['children', 'spouse', 'parents']:
    all_persons_train.extend(form_groups('data/train_'+predicate))

all_persons_test = []
for predicate in ['children', 'spouse', 'parents']:
    all_persons_test.extend(form_groups('data/test_'+predicate))


all_persons_unique_test = list(set(all_persons_test))
all_persons_unique_train = list(set(all_persons_train))


with open('init/persons_train', 'wb') as fp:
    pickle.dump(all_persons_unique_train, fp)

with open('init/persons_test', 'wb') as fp:
    pickle.dump(all_persons_unique_test, fp)

