import pickle

with open('log1.pickle', 'rb')as fp:
    data = pickle.load(fp)
print(data)