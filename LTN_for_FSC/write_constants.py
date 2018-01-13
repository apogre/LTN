
def form_groups(fname):
    all_persons = []
    predicate_vals = {'ethnicity':[],'religion':[],'cause_of_death':[],'institution':[], 'profession':[],\
                      'nationality':[], 'gender':[], 'spouse':[],'parents':[],'children':[], 'place_of_birth':[],\
                      'place_of_death':[],'location':[]}
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            predicate = persons[1]
            persons.pop(1)
            persons = [per.rstrip() for per in persons]
            persons = [per.replace('-', '_') for per in persons]
            if persons[1] not in predicate_vals[predicate]:
                predicate_vals[predicate].append(persons[1])
            all_persons.extend(persons)
    return all_persons, predicate_vals

all_persons_init, predicate_vals = form_groups('data/train.txt')

# for predicate in ['children', 'spouse', 'parents']:
#     all_persons_init.extend(form_groups('data/train_'+predicate))
#     all_persons_init.extend(form_groups('data/test_'+predicate))

all_persons_unique = list(set(all_persons_init))

for person in all_persons_unique:
    with open('persons_all.py', 'a') as f:
        const = person+' = ltn.Constant("'+person+'", domain=person)\n'
        f.write(const)
        update_list = 'everybody.append('+person+')\n'
        f.write(update_list)

for k,v in predicate_vals.iteritems():
    for val in v:
        with open('persons_all.py', 'a') as f:
            update_list = k+'_set.append('+val.rstrip()+')\n'
            f.write(update_list)