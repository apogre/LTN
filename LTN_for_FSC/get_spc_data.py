import sys

predicate = 'parents'


def get_data():
    count = 0
    with open("data/train.txt") as f:
        content = f.readlines()
        for line in content:
            if predicate in line:
                count += 1
                with open('data/'+predicate+'.txt', 'a') as the_file:
                    the_file.write(line)
        print count

# get_data()
# sys.exit(0)


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


all_persons_train = form_groups('data/train_spouse.txt')

all_persons_unique_train = list(set(all_persons_train))
# print len(all_persons)
# print len(all_persons_unique)

all_persons_unique_test = form_groups('data/test_spouse.txt')


# print all_persons_unique


#
# def process_spouse(predicate):
#     count = 0
#     with open('data/spouse_1') as f:
#         content = f.readlines()
#         for line in content:
#             persons = line.split('\t')
#             if persons[0].rstrip() not in all_persons_unique and persons[2].rstrip() not in all_persons_unique:
#                 count += 1
#                 with open('data/test_spouse.txt','a') as f:
#                     f.write(line)
#             else:
#                 with open('data/train_spouse.txt','a') as f:
#                     f.write(line)
#         print count

def process_train_test(predicate):
    count = 0
    with open('data/'+predicate+'.txt') as f:
        content = f.readlines()
        for line in content:
            persons = line.split('\t')
            if persons[0].rstrip() not in all_persons_unique_train and persons[2].rstrip() not in all_persons_unique_train:
                count += 1
                with open('data/test_'+predicate,'a') as f:
                    f.write(line)
            else:
                with open('data/train_'+predicate,'a') as f:
                    f.write(line)
        print count

process_train_test(predicate)