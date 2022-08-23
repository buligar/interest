import seaborn as sns
import sklearn.metrics as metrics
from scipy import stats # Библиотека для научных и технических вычислений.
from glob import glob # для работы с путями
from imblearn.over_sampling import RandomOverSampler
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from keras.models import Sequential # Для создания слоев
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # слои нейросети
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from keras.models import Sequential # Для создания слоев
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # слои нейросети
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,LabelEncoder # Стандартизируйте функции, удалив среднее значение и масштабируя до единичной дисперсии и LabelEncoder можно использовать для нормализации меток.
from sklearn.pipeline import make_pipeline, Pipeline # это служебная функция, которая является сокращением для построения конвейеров.
from sklearn.metrics import confusion_matrix # Вычислите матрицу путаницы, чтобы оценить точность классификации.
from sklearn.decomposition import PCA,NMF
from sklearn.feature_selection import SelectKBest,f_classif,SelectPercentile,RFECV,chi2 # Вычисление F-значение ANOVA для предоставленного образца и выбор функции в соответствии с процентилем наивысших оценок.
from sklearn.utils import resample # Распределение данных по различным классам
from sklearn.svm import LinearSVC
from PIL import Image # Библиотека для работы с изображениями



def HAM10000(self):
    # Загрузка обучающей выборки
    cell1 = pd.read_csv("csv/glcm_train0.csv", delimiter=';')
    cell1["class"] = 0
    cell2 = pd.read_csv("csv/glcm_train1.csv", delimiter=';')
    cell2["class"] = 1
    cell3 = pd.read_csv("csv/glcm_train2.csv", delimiter=';')
    cell3["class"] = 2
    cell4 = pd.read_csv("csv/glcm_train3.csv", delimiter=';')
    cell4["class"] = 3
    cell5 = pd.read_csv("csv/glcm_train4.csv", delimiter=';')
    cell5["class"] = 4
    cell6 = pd.read_csv("csv/glcm_train5.csv", delimiter=';')
    cell6["class"] = 5
    cell7 = pd.read_csv("csv/glcm_train6.csv", delimiter=';')
    cell7["class"] = 6
    cells = pd.concat([cell1, cell2, cell3, cell4, cell5, cell6, cell7])
    self.cells = cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

    # Загрузка тестовой выборки
    test_cell1 = pd.read_csv("csv/glcm_test0.csv", delimiter=';')
    test_cell1["class"] = 0
    test_cell2 = pd.read_csv("csv/glcm_test1.csv", delimiter=';')
    test_cell2["class"] = 1
    test_cell3 = pd.read_csv("csv/glcm_test2.csv", delimiter=';')
    test_cell3["class"] = 2
    test_cell4 = pd.read_csv("csv/glcm_test3.csv", delimiter=';')
    test_cell4["class"] = 3
    test_cell5 = pd.read_csv("csv/glcm_test4.csv", delimiter=';')
    test_cell5["class"] = 4
    test_cell6 = pd.read_csv("csv/glcm_test5.csv", delimiter=';')
    test_cell6["class"] = 5
    test_cell7 = pd.read_csv("csv/glcm_test6.csv", delimiter=';')
    test_cell7["class"] = 6
    test_cells = pd.concat([test_cell1, test_cell2, test_cell3, test_cell4, test_cell5, test_cell6, test_cell7])
    self.test_cells = test_cells.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

    # Соединение и преобразование выборок
    self.x_train = self.cells.drop(columns=self.cells.columns[-1]).to_numpy()
    self.y_train = cells.iloc[:, -1:].to_numpy().flatten()
    self.x_test = self.test_cells.drop(columns=self.test_cells.columns[-1]).to_numpy()
    self.y_test = test_cells.iloc[:, -1:].to_numpy().flatten()

    self.x_train = preprocessing.normalize(self.x_train)
    self.x_test = preprocessing.normalize(self.x_test)

def razm(self):
    self.UMCG()
    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            ("classify", LinearSVC(dual=False, max_iter=10000)),
        ]
    )

    N_FEATURES_OPTIONS = [2, 4, 8, 32]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            "reduce_dim": [PCA(), NMF()],
            "reduce_dim__n_components": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
        {
            "reduce_dim": [SelectKBest(chi2), SelectKBest(f_classif)],
            "reduce_dim__k": N_FEATURES_OPTIONS,
            "classify__C": C_OPTIONS,
        },
    ]
    reducer_labels = ["PCA", "NMF", "KBest(chi2)", "KBest(f_classif)"]

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
    grid.fit(self.x_train, self.y_train)

    mean_scores = np.array(grid.cv_results_["mean_test_score"])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + 0.5

    plt.figure()
    COLORS = "bgrcmyk"
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Сравнение методов сокращения признаков")
    plt.xlabel("Уменьшено количество функций")
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel("Точность классификации")
    plt.ylim((0, 1))
    plt.legend(loc="upper left")
    plt.show()

def glcm_test(self,k):
    # Очищение файла glcm_all.csv
    my_file = open("csv/glcm_all.csv", "w+")
    my_file.close()
    file_csv_path = 'csv/glcm_all.csv'
    os.remove(file_csv_path)
    self.features_sum = 8
    head = np.array(range(1, self.features_sum + 1)).flatten()
    num = len(self.files_path)  # Всего изображений
    self.num_cells = np.array(range(1, num + 1)).flatten()  # Массив кол-ва изображений
    df = pd.DataFrame(np.matrix(head))  # Создание заголовков столбцов
    df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f",
              sep=';')  # Указание кол-ва признаков
    select_segmentation = self.ui.select_segmentation.currentIndex()
    k += 1
    for i in range(num):
        self.file_path = self.files_path[i]
        print(self.file_path)
        self.segmentation_Unet()
        self.GLCM_RGB = []
        grayscale = cv2.imread(self.file_path,0)
        self.sqlite3_simple_pict_import(i, k)
        self.Distances = [self.spinBox.value()]
        glcm = greycomatrix(grayscale, distances=self.Distances,
                            angles=[0],
                            levels=256,
                            symmetric=True, normed=True)
        filt_glcm = glcm[1:, 1:, :, :]  # Не берет в расчет пиксель 0

        self.Contrast = greycoprops(filt_glcm, 'contrast')  # Текстурный признак Контраст
        self.Dissimilarity = greycoprops(filt_glcm, 'dissimilarity')  # Текстурный признак несходство
        self.Homogeneity = greycoprops(filt_glcm, 'homogeneity')  # Текстурный признак Локальная однородность
        self.Asm = greycoprops(filt_glcm, 'ASM')  # Текстурный признак Угловой второй момент
        self.Energy = greycoprops(filt_glcm, 'energy')  # Текстурный признак Энергия
        self.Correlation = greycoprops(filt_glcm, 'correlation')  # Текстурный признак Корреляция
        self.Entropy = greycoprops(filt_glcm, 'entropy')  # Текстурный признак Энтропия
        self.Max = greycoprops(filt_glcm, 'MAX')  # Текстурный признак Максимум вероятности

        # из двумерного массива в одномерный
        Contrast = np.concatenate(self.Contrast)
        Dissimilarity = np.concatenate(self.Dissimilarity)
        Homogeneity = np.concatenate(self.Homogeneity)
        Asm = np.concatenate(self.Asm)
        Energy = np.concatenate(self.Energy)
        Correlation = np.concatenate(self.Correlation)
        Entropy = np.concatenate(self.Entropy)
        Max = np.concatenate(self.Max)
        # Сохранение все вместе
        self.GLCM_All = [Contrast] + [Dissimilarity] + [Homogeneity] + [Asm] + [Energy] + [Correlation] + [
            Entropy] + [Max]
        self.GLCM_All = np.concatenate(self.GLCM_All)
        self.GLCM_RGB.append(self.GLCM_All)
        self.GLCM_RGB = np.concatenate(self.GLCM_RGB)
        mat = np.matrix(filt_glcm)
        df = pd.DataFrame(mat)
        df.to_csv('csv/glcm_all.csv', mode='a', header=False, index=False, float_format="%.5f", sep=';')
    print("Время выполнения работы функции расчета признаков с учетом параметров:")
@timer
def informativ2(self):  # Вычисление и отображение информативности признаков по отдельности
    np.set_printoptions(edgeitems=100000)
    # Проверка на указание расстояния смежности
    if self.spinBox.value() == 0:
        print("Укажите расстояние смежности в spinbox")
    self.rass = self.spinBox.value()  # Расстояние смежности
    self.rassinapravlenie = 4 * self.rass  # Расстояние смежности * 4 направления
    # Расчет информативности
    self.load2class() # Загрузка 2 классов обучающей выборки
    c = numpy.array(abs(self.mean1 - self.mean2))
    z = numpy.array(1.6 * (self.std1 + self.std2))
    self.informativeness = numpy.divide(c, z)
    infoContrast = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[0].reshape(self.rass, 4))
    infoDissimilation = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[1].reshape(self.rass, 4))
    infoHomogeneity = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[2].reshape(self.rass, 4))
    infoAsm = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[3].reshape(self.rass, 4))
    infoEnergy = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[4].reshape(self.rass, 4))
    infoCorrelation = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[5].reshape(self.rass, 4))
    infoEntropy = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[6].reshape(self.rass, 4))
    infoMax = np.transpose(self.informativeness.reshape(8, self.rassinapravlenie)[7].reshape(self.rass, 4))
    rass_zmez = np.arange(1, self.spinBox.value() + 1, 1)

    # Отоброжение на графиках
    plt.subplot(2, 4, 1)
    plt.grid(axis='both')
    plt.title("Зависимость информативности контраста\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoContrast[i], marker='o')
    plt.subplot(2, 4, 2)
    plt.grid(axis='both')
    plt.title("Зависимость информативности несходства\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoDissimilation[i], marker='o')
    plt.subplot(2, 4, 3)
    plt.grid(axis='both')
    plt.title("Зависимость информативности\n локальной однородности\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoHomogeneity[i], marker='o')
    plt.subplot(2, 4, 4)
    plt.grid(axis='both')
    plt.title("Зависимость информативности\n углового втрого момента\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoAsm[i], marker='o')
    plt.subplot(2, 4, 5)
    plt.grid(axis='both')
    plt.title("Зависимость информативности энергии\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoEnergy[i], marker='o')
    plt.subplot(2, 4, 6)
    plt.grid(axis='both')
    plt.title("Зависимость информативности корреляции\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoCorrelation[i], marker='o')
    plt.subplot(2, 4, 7)
    plt.grid(axis='both')
    plt.title("Зависимость информативности энтропии\n от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoEntropy[i], marker='o')
    plt.subplot(2, 4, 8)
    plt.grid(axis='both')
    plt.title("Зависимость информативности\n максимума вероятности от расстояния смежности")
    for i in range(4):
        plt.plot(rass_zmez, infoMax[i], marker='o')
    self.Anglelegend = [0, 45, 90, 135]
    plt.figlegend(self.Anglelegend)
    plt.show()
    print("Время выполнения работы функции информативность признаков по отдельности:")
@timer
def KNN(self):  # Классификатор k-ближайших соседей
    if self.spinBox_2.value() == 0:
        self.ui.output.setText('Укажите номер изображения')
    else:
        sample = self.ui.sample_box.currentIndex()
        if sample == 0:
            self.UMCG()
        elif sample == 1:
            self.HAM10000()
        results = {}
        for i in range(100):
            self.clf = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=i + 2)
            )
            self.clf.fit(self.x_train, self.y_train)
            results[i] = self.clf.score(self.x_test, self.y_test)
        acc = 0.001
        for v in results.items():
            if v > acc:
                acc = v
        print("Средняя точность по тестовой выборке:", acc)

        y_pred = []
        for i in range(len(self.x_test)):
            y_pred.append(self.clf.predict([self.x_test[i]]))
        y_pred = np.concatenate(y_pred)
        y_score = self.clf.predict_proba(self.x_test)[:, 1]
        Accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, fscore = score(self.y_test, y_pred)
        if sample == 0:
            ROC_AUC = roc_auc_score(self.y_test, y_pred)
        elif sample == 1:
            ROC_AUC = roc_auc_score(self.y_test, self.clf.predict_proba(self.x_test), multi_class='ovr')
        print("Точность:", Accuracy)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print("ROC_AUC: {}".format(ROC_AUC))
        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()
        # self.ui.accuracy.setNum(acc)  # отображение точности
        # self.num_test = self.spinBox_2.value() - 1  # Номер исследуемого изображения
        # res = self.clf.predict([self.x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
        # self.ui.results.setNum(res[0] + 1)  # отображение номера класса
        # self.show()  # показать на интерфейсе все значения
        print("Время выполнения работы функции KNN:")

@timer
def MLP(self):
    if self.spinBox_2.value()==0:
        self.ui.output.setText('Укажите номер изображения')
    else:
        sample = self.ui.sample_box.currentIndex()
        if sample == 0:
            self.UMCG()
        elif sample == 1:
            self.HAM10000()
        self.clf = MLPClassifier(random_state=1, max_iter=1000).fit(self.x_train, self.y_train)

        y_pred = []
        for i in range(len(self.x_test)):
            y_pred.append(self.clf.predict([self.x_test[i]]))
        y_pred = np.concatenate(y_pred)
        Accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, fscore, support = score(self.y_test, y_pred)
        if sample == 0:
            ROC_AUC = roc_auc_score(self.y_test, y_pred)
        elif sample ==1:
            ROC_AUC = roc_auc_score(self.y_test, self.clf.predict_proba(self.x_test), multi_class='ovr')
        print("Точность: {}".format(Accuracy))
        print('Прецизионность: {}'.format(precision))
        print('Отзыв: {}'.format(recall))
        print('F-мера: {}'.format(fscore))
        print('Поддержка: {}'.format(support))
        print("ROC_AUC: {}".format(ROC_AUC))
        y_score = self.clf.predict_proba(self.x_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()
        # self.ui.accuracy.setNum(results)  # отображение точности
        # self.num_test = self.spinBox_2.value() - 1  # Номер исследуемого изображения
        # res = self.clf.predict([self.x_test[self.num_test]])  # предсказание класса для выбранного изображения из тестовой выборки
        # self.ui.results.setNum(res[0] + 1)  # отображение номера класса
        # self.show()  # показать на интерфейсе все значения
        print("Время выполнения работы функции MLP:")

@timer
def SVM(self):
    sample = self.ui.sample_box.currentIndex()
    feature_selection = self.ui.feature_selection.currentIndex()
    if sample == 0:
        self.UMCG()
    elif sample == 1:
        self.HAM10000()
    self.features_sum = 128 * self.spinBox.value()
    if feature_selection == 1:
        self.features_sum = self.len_max_features
    if sample == 1:
        my_file = open("csv/glcm_all_7.csv", "w+")
        my_file.close()
        file_csv_path = 'csv/glcm_all_7.csv'
        os.remove(file_csv_path)
        mas = []
        for i in range(1, self.features_sum+1):
            mas.append(str(i))
        head=mas+["class"]
        df = pd.DataFrame(np.matrix(head))  # Создание заголовков столбцов
        df.to_csv('csv/glcm_all_7.csv', mode='a', header=False, index=False, float_format="%.5f",
                  sep=',')  # Указание кол-ва признаков
        df = self.cells.append(self.test_cells)
        print(df)
        df.to_csv('csv/glcm_all_7.csv', mode='a', header=False, index=False, float_format="%.5f", sep=',')
        dataset_images_RGB = pd.read_csv("csv/glcm_all_7.csv")
        images = dataset_images_RGB.drop(['class'], axis=1)
        labels = dataset_images_RGB['class']
        print(images)
        print(labels)
        # Oversampling to overcome class imbalance
        oversample = RandomOverSampler()
        images, labels = oversample.fit_resample(images, labels)
        print(images.shape)

        # Keeping a smaller sample so that the cross-validation doesn't take too long
        images = images.sample(frac=1, random_state=1,replace=True)
        labels = labels.sample(frac=1, random_state=1,replace=True)
        print(images.shape)
        # restructuring the images to be fitted in the model
        images = images.astype('float32')
        # Normalizing the images.
        images = (images - np.mean(images,axis=0)) / np.std(images)
        # Splitting my predictive and response data into training and testing sets with an 80:20 ratio
        # while the state is set to a constant so that the splitting can be done reproducibly
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            images, labels, random_state=1, test_size=0.20)

    # Performing LDA for dimentionality reduction
    num_classes = 7
    lda = LDA()
    self.x_train = lda.fit_transform(self.x_train, self.y_train)
    self.x_test = lda.transform(self.x_test)

    start = time.time()

    # Finding the best parameters by cross-validation
    parameters = [{'kernel': ['rbf'],
                   'gamma': [0.01, 0.1, 0.5],
                   'C': [10, 100, 1000]}]
    print("# Tuning hyper-parameters")
    clf = GridSearchCV(SVC(), parameters, cv=num_classes)
    clf.fit(self.x_train, self.y_train)

    print('best parameters:')
    print(clf.best_params_)
    print('-------------------------------------')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    stop = time.time()
    # setting the optimal parameters that were found
    optimal_C = 10
    optimal_gamma = 0.01

    # Fitting the model
    svc = SVC(kernel="rbf", gamma=optimal_gamma, C=optimal_C)
    svc.fit(self.x_train, self.y_train)
    pred = svc.predict(self.x_test)
    # printing the accuracy of the SVM model
    print("Оценка точности: ", accuracy_score(self.y_test, pred))

    print("Время построения и обучения модели составляет : ", (stop - start) / 60, " minutes")

    # Setting up the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true=self.y_test, y_pred=pred)

    # plotting the confusion matrix for the model label prediction
    ax = sns.heatmap(confusion_matrix, fmt='', cmap='Blues')
    ax.set_title('Матрица путаницы с метками\n');
    ax.set_xlabel('Прогнозируемые метки')
    ax.set_ylabel('Фактические этикетки')
    plt.show()

    # plotting the incorrect prediction fraction of each class label
    label_frac_error = 1 - np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    plt.bar(np.arange(7), label_frac_error)
    plt.title('Неверное предсказание доли меток')
    plt.xlabel('Настоящая этикетка')
    plt.ylabel('Неправильно классифицирована дробь')
    plt.show()

def NN(self):

    # Настройка параметров
    np.random.seed(42)
    SIZE = self.image_size.value()
    batch_size = self.ui.batch_size.currentIndex()
    if batch_size == 0:
        batch_size = 16
    else:
        batch_size = 32
    epochs = self.ui.epochs.value()
    n_samples = self.image_norm.value()
    print(SIZE)
    print(batch_size)
    print(epochs)
    print(n_samples)

    # кодирование метки в числовые значения из текста
    skin_df = pd.read_csv(self.metadata)
    le = LabelEncoder()
    le.fit(skin_df['dx'])
    LabelEncoder()
    # print(list(le.classes_))

    skin_df['label'] = le.transform(skin_df["dx"])
    # print(skin_df.sample(10))

    # Визуализация распределения данных
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221)
    skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Count')
    ax1.set_title('Cell Type');

    ax2 = fig.add_subplot(222)
    skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_ylabel('Count', size=15)
    ax2.set_title('Sex');

    ax3 = fig.add_subplot(223)
    skin_df['localization'].value_counts().plot(kind='bar')
    ax3.set_ylabel('Count', size=12)
    ax3.set_title('Localization')

    ax4 = fig.add_subplot(224)
    sample_age = skin_df[pd.notnull(skin_df['age'])]
    sns.distplot(sample_age['age'], fit=stats.norm, color='red');
    ax4.set_title('Age')

    plt.tight_layout()
    plt.show()

    # Данные баланса.
    # Много способов сбалансировать данные... вы также можете попробовать назначить веса во время model.fit
    # Разделяем каждый класс, передискретизируем и объединяем обратно в один фрейм данных

    df_0 = skin_df[skin_df['label'] == 0]
    df_1 = skin_df[skin_df['label'] == 1]
    df_2 = skin_df[skin_df['label'] == 2]
    df_3 = skin_df[skin_df['label'] == 3]
    df_4 = skin_df[skin_df['label'] == 4]
    df_5 = skin_df[skin_df['label'] == 5]
    df_6 = skin_df[skin_df['label'] == 6]


    df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
    df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
    df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
    df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
    df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
    df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
    df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

    # Объединяем обратно в один фрейм данных
    skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                                  df_2_balanced, df_3_balanced,
                                  df_4_balanced, df_5_balanced, df_6_balanced])

    # Теперь пришло время прочитать изображения на основе идентификатора изображения из CSV-файла.
    # Это самый безопасный способ чтения изображений, поскольку он гарантирует, что правильное изображение будет прочитано для правильного идентификатора
    image_path = {os.path.splitext(os.path.basename(x))[0]: x
                  for x in glob(os.path.join(self.train_papka, '*', '*.jpg'))}

    # Определяем путь и добавляем как новый столбец
    skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
    # Используйте путь для чтения изображений.
    skin_df_balanced['image'] = skin_df_balanced['path'].map(
        lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))
    print(skin_df_balanced)

    # Преобразование столбца dataframe изображений в массив numpy
    X = np.asarray(skin_df_balanced['image'].tolist())
    X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
    Y = skin_df_balanced['label']  # Assign label values to Y
    Y_cat = to_categorical(Y,num_classes=7)  # Convert to categorical as this is a multiclass classification problem
    # Разделить на обучение и тестирование
    x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)
    print(x_train)
    # Определить модель.
    # Я использовал autokeras, чтобы найти наилучшую модель для этой задачи.
    # Вы также можете загрузить предварительно обученные сети, такие как мобильная сеть или VGG16.

    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(32))
    model.add(Dense(7, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

    # Обучение
    # Вы также можете использовать генератор, чтобы использовать аугментацию во время обучения.

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=2)

    model.save(f"models/model{SIZE}_imagesize{n_samples}_n_samples_{epochs}Epoch.h5")
    fig = plt.figure(figsize=(12, 8))
    # отображать точность обучения и проверки и потери в каждую эпоху
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Прогноз на тестовых данных
    y_pred = model.predict(x_test)
    # Преобразование классов прогнозов в один горячий вектор
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Преобразование тестовых данных в один горячий вектор
    y_true = np.argmax(y_test, axis=1)

    # Вывести матрицу путаницы
    cm = confusion_matrix(y_true, y_pred_classes)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

    # PLot дробно-неправильных классификаций
    incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
    plt.bar(np.arange(7), incorr_fraction)
    plt.xlabel('True Label')
    plt.ylabel('Fraction of incorrect predictions')

def filters(self):
    self.choosefilter = self.ui.filterbox.currentIndex()
    self.blur = cv2.imread(self.file_path)  # Преобразование в оттенки серого
    if self.choosefilter == 1:
        self.blur = cv2.blur(self.blur, (5,5))
    elif self.choosefilter == 2:
        self.blur = cv2.GaussianBlur(self.blur, (5, 5), 0)
    elif self.choosefilter == 3:
        self.blur = cv2.medianBlur(self.blur, 5)
    elif self.choosefilter == 4:
        self.blur = cv2.bilateralFilter(self.blur, 9, 75, 75)
    cv2.imwrite("blurred.png", self.blur)

def tpot(self):
    """
    Вход: x_train, y_train, x_test, y_test

    :return: лучшая модель классификации tpot
    """
    self.UMCG()
    tpot = TPOTClassifier(warm_start=True,verbosity=2)
    tpot.fit(self.x_train, self.y_train)
    print(tpot.score(self.x_test, self.y_test))
    tpot.export('tpot_digits_pipeline.py')
    print("Время выбора лучшей модели:")

def deletebackground(self): # Удаления фона

    img = cv2.imread(self.file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # преобразовать в серый цвет
    # порог
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # применяем морфологию для очистки небольших пятен
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # получаем внешний контур
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    # рисуем белый залитый контур на черном фоне как мас
    contour = np.zeros_like(gray)
    cv2.drawContours(contour, [big_contour], 0, 255, -1)
    # размытие увеличить изображение
    blur = cv2.GaussianBlur(contour, (5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    # растянуть так, чтобы 255 -> 255 и 127,5 -> 0
    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255))
    # поместить маску в альфа-канал ввода
    self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    self.result[:, :, 3] = mask
    # сохранить вывод
    cv2.imwrite('withoutBackground.png', self.result)
    # Вывод на экран
    self.scene_2.clear()
    self.pixmap2 = QPixmap('withoutBackground.png')
    self.scene_2.addPixmap(self.pixmap2)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Время выполнения работы функции удаление фона:")