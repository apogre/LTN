
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

all_persons_init = []

for predicate in ['children', 'spouse', 'parents']:
    all_persons_init.extend(form_groups('data/train_'+predicate))
    all_persons_init.extend(form_groups('data/test_'+predicate))


all_persons_unique = list(set(all_persons_init))


for person in all_persons_unique:
    with open('init/persons.txt', 'a') as f:
        const = person+' = ltn.Constant("'+person+'", domain=person)\n'
        f.write(const)