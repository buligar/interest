import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression # логистическая регрессия
from sklearn.neighbors import KNeighborsClassifier # к-ближайших соседей
from sklearn.model_selection import GridSearchCV,StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler # подключение преобразователя данных
from sklearn.pipeline import make_pipeline # подключение сокращения для конструктора конвейера
from sklearn.model_selection import RepeatedStratifiedKFold


np.set_printoptions(edgeitems=100000)
massive=np.array(range(112,125)).flatten()
print(massive)
train_m=np.array(range(0,1360)).flatten()
test_m = np.array(range(1360,1699)).flatten()
dataset = pd.read_csv('Myocardial infarction complications Database.csv', delimiter=',')
# dataset.to_csv('myo.csv', index=False, float_format="%.5f", sep=';') # Указание кол-ва признак
# dataset = pd.read_csv('myo.csv', delimiter=';')
print(dataset)
class_dataset = dataset["FK_STENOK"]
dataset["class"] = class_dataset
dataset = dataset.fillna(0)
dataset = dataset.astype({"class": "Int64"})
features_train = dataset.iloc[train_m, massive]
features_test = dataset.iloc[test_m, massive]

print(features_train)

train_cells = features_train.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))
test_cells = features_test.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','.'), errors='coerce'))

x_train = train_cells.drop(columns=train_cells.columns[-1]).to_numpy()
y_train = train_cells.iloc[:,-1:].to_numpy().flatten()
x_test = test_cells.drop(columns=test_cells.columns[-1]).to_numpy()
y_test = test_cells.iloc[:,-1:].to_numpy().flatten()

# # Размеры
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
#
# # Вид
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

### Простой KNN ###
model = KNeighborsClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("KNN:",score)

### Простой LR ###
model = LogisticRegression()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("LR",score)


### Чуть лучше KNN (перебирает несколько параметров) ###
results = {}
for i in range(100):
    neigh_pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=i+2)
    )
    neigh_pipe.fit(x_train, y_train)
    results[i] = neigh_pipe.score(x_test, y_test)
acc = 0.001
n_neighbors = 0
for k, v in results.items():
    if v > acc:
        acc = v
        n_neighbors = k
print("Средняя точность по тестовой выборке:", acc)
print("Оптимальное количество соседей:", n_neighbors)


### Чуть лучше LR (перебирает несколько параметров) ###
def get_models():
	models = dict()
	for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
		# create name for model
		key = '%.4f' % p
		# turn off penalty in some cases
		if p == 0.0:
			# no penalty in this case
			models[key] = LogisticRegression(multi_class='multinomial',max_iter=1000, solver='lbfgs', penalty='none')
		else:
			models[key] = LogisticRegression(multi_class='multinomial',max_iter=1000, solver='lbfgs', penalty='l2', C=p)
	return models

def evaluate_model(model, x_train, y_train):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

models = get_models()
results, names = list(), list()

for name, model in models.items():
	# evaluate the model and collect the scores
	scores = evaluate_model(model, x_train, y_train)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize progress along the way
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()