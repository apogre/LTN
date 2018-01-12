

def form_groups(fname):
    spouse_of = {}
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            persons = [per.rstrip() for per in persons]
            if persons[2] not in spouse_of.keys():
                spouse_of[persons[2]] = [persons[0]]
            else:
                spouse_of[persons[2]].append(persons[0])
    return spouse_of


predicate = 'spouse'

spouse_of = form_groups('data/'+predicate)


def get_all_persons(fname):
    all_persons = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            persons = [per.rstrip() for per in persons]
            persons.pop(1)
            all_persons.extend(persons)
    return all_persons


all_persons_init = []

for predicate in ['children', 'spouse', 'parents']:
    all_persons_init.extend(get_all_persons('data/train_'+predicate))
    all_persons_init.extend(get_all_persons('data/test_'+predicate))


all_persons_unique = list(set(all_persons_init))

for person in all_persons_unique:
    if person not in spouse_of.keys():
        spouse_of[person] = set()

