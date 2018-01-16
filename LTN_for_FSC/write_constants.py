# predicate_vals = {'ethnicity': [], 'religion': [], 'cause_of_death': [], 'institution': [], 'profession': [], \
#                   'nationality': [], 'gender': [], 'spouse': [], 'parents': [], 'children': [], 'place_of_birth': [], \
#                   'place_of_death': [], 'location': []}

predicate_vals = {'spouse': [], 'parents': [], 'children': []}


def form_groups(fname, predicate_vals):
    all_persons = []
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

all_persons_init, predicate_vals = form_groups('train_freebase_sample.txt', predicate_vals)

all_persons_unique = list(set(all_persons_init))

for person in all_persons_unique:
    with open('persons_all_spc_sample.py', 'a') as f:
        const = person+' = ltn.Constant("'+person+'", domain=person)\n'
        f.write(const)
        update_list = 'first_group.append('+person+')\n'
        f.write(update_list)

predicate_vals = {'spouse': [], 'parents': [], 'children': []}

all_persons_init_test, predicate_vals_test = form_groups('test_freebase_sample.txt', predicate_vals)

all_persons_unique_test = list(set(all_persons_init_test))

for person in all_persons_unique_test:
    if person not in all_persons_unique:
        with open('persons_all_spc_sample.py', 'a') as f:
            const = person+' = ltn.Constant("'+person+'", domain=person)\n'
            f.write(const)
            update_list = 'second_group.append('+person+')\n'
            f.write(update_list)
    else:
        print person

print len(all_persons_unique)
print len(all_persons_unique_test)
# for k, v in predicate_vals.iteritems():
#     for val in v:
#         with open('persons_all_spc.py', 'a') as f:
#             update_list = k+'_set.append('+val.rstrip()+')\n'
#             f.write(update_list)