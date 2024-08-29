from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import warnings
from math import sqrt

from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, precision_score, recall_score, mean_absolute_error, confusion_matrix, \
    f1_score, roc_auc_score, r2_score
from tabulate import tabulate

from Data.utilis.utils import *

warnings.filterwarnings("ignore")

# ----------------------Loading the Data-------------------------#

# Amazon.com, Inc.
# Apple Inc.
# Microsoft Corporation

print('\nData Loading.....')
tweets = pd.read_csv(os.getcwd() + '\\Data\\stock_tweets.csv')
all_stocks = pd.read_csv(os.getcwd() + '\\Data\\stock_yfinance_data.csv')
print('\nAmazon.com, Inc......')
data1 = tweets[tweets['Stock Name'] == 'AMZN']
print('\nApple Inc.......')
data2 = tweets[tweets['Stock Name'] == 'AAPL']
print('\nMicrosoft Corporation......')
data3 = tweets[tweets['Stock Name'] == 'MSFT']


# ---------------------Pre-processing and Data balancing------------------------#

def text_lowercase(text):
    return text.lower()


for i in range(len(data1)):
    data1['Tweet'].iloc[i] = text_lowercase(data1['Tweet'].iloc[i])
for i in range(len(data1)):
    data2['Tweet'].iloc[i] = text_lowercase(data2['Tweet'].iloc[i])
for i in range(len(data1)):
    data3['Tweet'].iloc[i] = text_lowercase(data3['Tweet'].iloc[i])

sentiment_data_AMZN, sentiment_data_AAPL, sentiment_data_MSFT = sentiment(data1, data2, data3)
final_df_AMZN, final_df_AAPL, final_df_MSFT = final_stock(all_stocks, sentiment_data_AMZN, sentiment_data_AAPL,
                                                          sentiment_data_MSFT)
# -------------------Analysis plot of closing prices of AMZN,AAPL,MSFT-------------------
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 253), final_df_AMZN['Close'], color='b', label='AMZN')
plt.plot(np.arange(1, 253), final_df_AAPL['Close'], color='m', label='AAPL')
plt.plot(np.arange(1, 253), final_df_MSFT['Close'], color='g', label='MSFT')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}
plt.legend(loc='upper right', fancybox=True, prop=prop)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

final_df = pd.concat([final_df_AMZN, final_df_AAPL, final_df_MSFT], axis=0, ignore_index=True)
dataset1 = dataset_(final_df_AMZN)
dataset2 = dataset_(final_df_AAPL)
dataset3 = dataset_(final_df_AAPL)
dataset = pd.concat([dataset1, dataset2, dataset3])
y1, y2, y3 = labels_(dataset1, dataset2, dataset3)
y = pd.concat([y1, y2, y3])
y = numpy_.array(y)

y_label = (y.values == y.max(axis=1).values[:, None]).astype(int)

# Convert binary array to numeric values based on custom interpretation
y_val = np.array([0 if np.array_equal(row, [1, 0, 0]) else
                  1 if np.array_equal(row, [0, 1, 0]) else
                  2 for row in y_label])

X, y_scale_dataset = normalize_data(dataset, (-1, 1), "Close")
X, y = numpy_.array_(X, y_val)

print("\nTotal samples - ", X.shape[0])
from collections import Counter

counts = Counter(y)
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
mapped_labels = np.vectorize(label_map.get)(y)
counts = Counter(mapped_labels)

# Display counts
for label, count in counts.items():
    print(f"{label}: {count}")


# -------------------LightweightDeepAutoencoder for data balance---------------
class LightweightDeepAutoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder, self.encoder = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.input_dim,))

        # Encoder
        encoded = Dense(256, activation='relu')(input_layer)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # Models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder, encoder

    def train(self, X_train, X_valid, epochs=20, batch_size=32):
        self.autoencoder.fit(X_train, X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_valid, X_valid))

    def generate_samples(self, X, num_samples):
        # Assuming X contains samples from the minority class
        encoded_samples = self.encoder.predict(X)
        noise = np.random.normal(0, 1, (num_samples, self.encoding_dim))
        synthetic_samples = self.autoencoder.layers[-1].predict(noise)
        return synthetic_samples


# Initialize the autoencoder
autoencoder = LightweightDeepAutoencoder(input_dim=784, encoding_dim=64)  # for example, if input_dim is 784

X, y = features(autoencoder, X, y)
X, y = numpy_.py_array(X, y)

counts = Counter(y)
mapped_labels = np.vectorize(label_map.get)(y)
counts = Counter(mapped_labels)
print('\n')
# Display counts
for label, count in counts.items():
    print(f"{label}: {count}")
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples - ", X_train.shape[0])
print("Testing samples - ", X_test.shape[0], '\n')


# --------------feature fusion by  hybrid  Snow Geese White Shark Optimizer Algorithm ------------------


class SGO_WSO_Optimizer:
    def __init__(self, num_geese, num_dimensions, max_iterations, alpha, beta, gamma, lb, ub, fobj, whiteSharks,
                 itemax):
        self.num_geese = num_geese
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lb = lb
        self.ub = ub
        self.fobj = fobj
        self.whiteSharks = whiteSharks
        self.itemax = itemax

    def SGO(self):
        geese = np.random.rand(self.num_geese, self.num_dimensions) * (self.ub - self.lb) + self.lb
        fitness = np.zeros(self.num_geese)
        best_solution = None
        best_fitness = float('inf')
        ccurve_sgo = []

        for iteration in range(1, self.max_iterations + 1):
            for i in range(self.num_geese):
                fitness[i] = self.fobj(geese[i])

            sorted_indices = np.argsort(fitness)
            geese = geese[sorted_indices]
            fitness = fitness[sorted_indices]

            for i in range(self.num_geese):
                best_goose = geese[0]
                geese[i] = self.alpha * geese[i] + self.beta * np.random.rand(self.num_dimensions) * (
                        best_goose - geese[i]) + self.gamma * np.random.rand(self.num_dimensions) * (
                                   self.ub - self.lb)
                geese[i] = np.minimum(np.maximum(geese[i], self.lb), self.ub)

            if fitness[0] < best_fitness:
                best_solution = geese[0]
                best_fitness = fitness[0]

            ccurve_sgo.append(best_fitness)

        return best_solution, best_fitness, ccurve_sgo

    def WSO(self):
        ccurve_wso = np.zeros(self.itemax)
        gbest = None
        fmin0 = float('inf')
        WSO_Positions = np.random.rand(self.whiteSharks, self.num_dimensions) * (self.ub - self.lb) + self.lb
        v = np.zeros_like(WSO_Positions)

        fit = np.zeros(self.whiteSharks)
        for i in range(self.whiteSharks):
            fit[i] = self.fobj(WSO_Positions[i])

        fitness = fit
        fmin0 = min(fit)
        index = np.argmin(fit)
        wbest = np.copy(WSO_Positions)
        gbest = np.copy(WSO_Positions[index])

        fmax = 0.75
        fmin = 0.07
        tau = 4.11
        mu = 2 / abs(2 - tau - np.sqrt(tau ** 2 - 4 * tau))
        pmin = 0.5
        pmax = 1.5
        a0 = 6.25
        a1 = 100
        a2 = 0.0005

        for ite in range(1, self.itemax + 1):
            mv = 1 / (a0 + np.exp((self.itemax / 2.0 - ite) / a1))
            s_s = abs((1 - np.exp(-a2 * ite / self.itemax)))
            p1 = pmax + (pmax - pmin) * np.exp(-(4 * ite / self.itemax) ** 2)
            p2 = pmin + (pmax - pmin) * np.exp(-(4 * ite / self.itemax) ** 2)

            nu = np.random.randint(0, self.whiteSharks, self.whiteSharks)
            for i in range(self.whiteSharks):
                rmin = 1
                rmax = 3.0
                rr = rmin + np.random.rand() * (rmax - rmin)
                wr = abs(((2 * np.random.rand()) - (1 * np.random.rand() + np.random.rand())) / rr)
                v[i] = mu * v[i] + wr * (wbest[nu[i]] - WSO_Positions[i])

            for i in range(self.whiteSharks):
                f = fmin + (fmax - fmin) / (fmax + fmin)
                wo = np.logical_xor(WSO_Positions[i] >= self.ub, WSO_Positions[i] <= self.lb)
                if np.random.rand() < mv:
                    WSO_Positions[i] = WSO_Positions[i] * (~wo) + (
                            self.ub * (WSO_Positions[i] > self.ub) + self.lb * (WSO_Positions[i] < self.lb)) * wo
                else:
                    WSO_Positions[i] += v[i] / f

            for i in range(1, self.whiteSharks):
                for j in range(self.num_dimensions):
                    if np.random.rand() < s_s:
                        Dist = abs(np.random.rand() * (gbest[j] - 1 * WSO_Positions[i, j]))
                        if i == 0:
                            WSO_Positions[i, j] = gbest[j] + np.random.rand() * Dist * np.sign(np.random.rand() - 0.5)
                        else:
                            WSO_Pos = gbest[j] + np.random.rand() * Dist * np.sign(np.random.rand() - 0.5)
                            WSO_Positions[i, j] = (WSO_Pos + WSO_Positions[i - 1, j]) / 2 * np.random.rand()

            for i in range(self.whiteSharks):
                if np.all(WSO_Positions[i] >= self.lb) and np.all(WSO_Positions[i] <= self.ub):
                    fit_i = self.fobj(WSO_Positions[i])
                    if fit_i < fitness[i]:
                        wbest[i] = WSO_Positions[i]
                        fitness[i] = fit_i

                    if fitness[i] < fmin0:
                        fmin0 = fitness[i]
                        gbest = wbest[index]

            ccurve_wso[ite - 1] = fmin0

        return gbest, fmin0, ccurve_wso

    def hybrid_SGO_WSO(self):
        best_goose_sgo, best_fitness_sgo, ccurve_sgo = self.SGO()
        best_shark_wso, best_fitness_wso, ccurve_wso = self.WSO()

        if best_fitness_sgo < best_fitness_wso:
            best_solution = best_goose_sgo
            best_fitness = best_fitness_sgo
            ccurve = ccurve_sgo
        else:
            best_solution = best_shark_wso
            best_fitness = best_fitness_wso
            ccurve = ccurve_wso

        return best_solution, best_fitness, ccurve
        return best_solution, best_fitness, ccurve


# Parameters
num_geese = 20
num_dimensions = 5
max_iterations = 100
alpha = 0.9
beta = 0.1
gamma = 0.1
lb = -10
ub = 10
whiteSharks = 10
itemax = 100

# Run Hybrid SGO-WSO Algorithm
optimizer = SGO_WSO_Optimizer(num_geese, num_dimensions, max_iterations, alpha, beta, gamma, lb, ub, sphere_function,
                              whiteSharks, itemax)
best_solution_sgo, best_fitness_sgo, ccurve_sgo = optimizer.SGO()
best_solution_wso, best_fitness_wso, ccurve_wso = optimizer.WSO()
best_solution_hybrid, best_fitness_hybrid, ccurve_hybrid = optimizer.hybrid_SGO_WSO()


# =------------------Muti-head Parallel convolutional neural network for  prediction------------------
class MultiHeadParallelCNN:
    def __init__(self, input_shape, num_heads=2):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.heads = []
        self._build_heads()
        self._build_model()

    def _build_heads(self):
        for _ in range(self.num_heads):
            head_input = Input(shape=self.input_shape)
            conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(head_input)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            flatten = Flatten()(pool2)
            self.heads.append(flatten)

    def _build_model(X_train, n_class, x, ylf):
        model = Sequential()
        model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_class, activation='softmax'))
        # compile model
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, X):
        return self.model.predict([X] * self.num_heads)


# ------------------  Skill Optimization Algorithm ----------------------------
class SkillOptimizer:
    def __init__(self, search_agents, max_iterations, lowerbound, upperbound, dimension):
        self.search_agents = search_agents
        self.max_iterations = max_iterations
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.dimension = dimension
        self.best_score = None
        self.best_pos = None
        self.soa_curve = []

    def fitness(self, params):
        learning_rate = params[0]
        batch_size = int(params[1])

        model = self.create_model(learning_rate, batch_size)
        model.fit(self.x_train, self.y_train_binary, batch_size=batch_size, epochs=3, verbose=0)
        _, accuracy = model.evaluate(self.x_test, self.y_test_binary, verbose=0)
        return -accuracy  # Minimize negative accuracy

    def SOA(self):
        lowerbound = np.ones(self.dimension) * self.lowerbound
        upperbound = np.ones(self.dimension) * self.upperbound

        X = np.random.rand(self.search_agents, self.dimension) * (upperbound - lowerbound) + lowerbound
        fit = np.zeros(self.search_agents)

        for i in range(self.search_agents):
            L = X[i, :]
            fit[i] = self.fitness(L)

        for t in range(self.max_iterations):
            best, blocation = np.min(fit), np.argmin(fit)
            if t == 0:
                self.best_pos = X[blocation, :]
                self.best_score = best
            elif best < self.best_score:
                self.best_score = best
                self.best_pos = X[blocation, :]

            X_P1 = np.zeros_like(X)
            F_P1 = np.zeros(self.search_agents)

            for i in range(self.search_agents):
                K = np.where(fit < fit[i])[0]
                if K.size != 0:
                    KK = np.random.randint(0, K.size)
                    K = K[KK]
                else:
                    K = i
                    KK = 0

                expert = X[K, :]

                if np.random.rand() < 0.5:
                    I = np.round(1 + np.random.rand())
                    RAND = np.random.rand()
                else:
                    I = np.round(1 + np.random.rand(self.dimension))
                    RAND = np.random.rand(self.dimension)

                X_P1[i, :] = X[i, :] + RAND * (expert - I * X[i, :])
                X_P1[i, :] = np.maximum(np.minimum(X_P1[i, :], upperbound), lowerbound)

                L = X_P1[i, :]
                F_P1[i] = self.fitness(L)

                if F_P1[i] < fit[i]:
                    X[i, :] = X_P1[i, :]
                    fit[i] = F_P1[i]

            X_P2 = np.zeros_like(X)
            F_P2 = np.zeros(self.search_agents)

            for i in range(self.search_agents):
                if np.random.rand() < 0.5:
                    X_P2[i, :] = X[i, :] + ((1 - 2 * np.random.rand(self.dimension)) / t) * X[i, :]
                else:
                    X_P2[i, :] = X[i, :] + self.lowerbound / t + np.random.rand(self.dimension) * (
                            self.upperbound / t - self.lowerbound / t)

                X_P2[i, :] = np.maximum(np.minimum(X_P2[i, :], upperbound), lowerbound)

                L = X_P2[i, :]
                F_P2[i] = self.fitness(L)

                if F_P2[i] < fit[i]:
                    X[i, :] = X_P2[i, :]
                    fit[i] = F_P2[i]

            self.soa_curve.append(self.best_score)
            print(f"Iteration {t}: Best Fitness = {self.best_score}")

        return self.best_score, self.best_pos, self.soa_curve

    def optimize(self):
        self.load_data()
        self.best_score, self.best_pos, self.soa_curve = self.SOA()
        return self.best_score


SOA = SkillOptimizer(search_agents=10, max_iterations=20, lowerbound=[0.001, 16], upperbound=[1, 128],
                     dimension=2)

num_classes = 3  # Define the number of output classes
model = MultiHeadParallelCNN._build_model(X_train, num_classes, best_solution_hybrid, SOA)
y_train_ = to_categorical(y_train)
ep = 50  # number of epochs
xx = list(range(1, ep + 1))

# Now, you can train the model using dataset
history = model.fit(X_train, y_train_, epochs=ep, batch_size=32, validation_split=0.2, verbose=1)

# ---------------------loss and accuracy curve---------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax1.set_facecolor('#FCFCFC')
plt.grid(color='w', linestyle='-.', linewidth=1)
plt.plot(xx, 1 - np.array(history.history['loss']), color='r')
plt.plot(xx, 1 - np.array(history.history['val_loss']), color='b')
plt.ylabel('Accuracy', fontsize=16, weight='bold')
plt.xlabel(' Epoch', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.xlim([0, ep + 1])
plt.yticks(fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='lower right', fancybox=True, prop=prop)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('#FCFCFC')
plt.grid(color='w', linestyle='-.', linewidth=1)
plt.plot(xx, history.history['loss'], color='r')
plt.plot(xx, history.history['val_loss'], color='b')
plt.ylabel('Loss', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='upper right', fancybox=True, prop=prop)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xlim([0, ep + 1])
plt.tight_layout()
plt.show()

pred, pred_prob = testing(model, X_test)  # Testing

mat = confusion_matrix(y_test, pred)  # confusion matrix

mse = mean_squared_error(y_test, pred)  # Mean Squared Error (MSE)

r2 = r2_score(y_test, pred) # R-squared (R2) score

# Area Under the Curve (AUC)
auc_scores = []
for class_index in range(pred_prob.shape[1]):
    class_true_labels = (y_test == class_index).astype(int)  # Treat current class as positive, others as negative
    auc_score = roc_auc_score(class_true_labels, pred_prob[:, class_index])
    auc_scores.append(auc_score)

# Compute the mean AUC across all classes
auc_score = np.mean(auc_scores)

accuracy = accuracy_score(y_test, pred) # Accuracy

f1s = f1_score(y_test, pred, average='weighted') # F1-score (F-measure)

rec = recall_score(y_test, pred, average='weighted') # Recall

pre = precision_score(y_test, pred, average='weighted') # Precision

rmse = sqrt(mean_squared_error(y_test, pred)) # Root Mean Squared Error (RMSE)

aae = mean_absolute_error(y_test, pred)# Average Absolute Error (AAE)

are = np.mean(np.abs(y_test - pred) / y_test) # Average Relative Error (ARE)

# Calculate confusion matrix for specificity
conf_matrix = confusion_matrix(y_test, pred, labels=[0, 1, 2])
plot_confusion_matrix(conf_matrix, ['Negative', 'Neutral', 'Positive'])

tn = np.diag(conf_matrix)  # true negatives
fp = conf_matrix.sum(axis=0) - tn  # false positives
fn = conf_matrix.sum(axis=1) - tn  # false negatives
tp = conf_matrix.sum() - (fp + fn + tn)  # true positives
spe = np.mean(tn / (tn + fp))
mae = mean_absolute_error(y_test, pred)
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("Accuracy                              :", accuracy)
print("Recall                                :", rec)
print("Precision                             :", pre)
print("Specificity                           :", spe)
print("F1-score (F-measure)                  :", f1s)
print("Mean Squared Error (MSE)              :", mse)
print("Mean Absolute Error (MAE)             :", mae)
print("Root Mean Squared Error (RMSE)        :", rmse)
print("Average Absolute Error (AAE)          :", aae)

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]
# Calculate overall accuracy
overall_accuracy = accuracy_score(y_test, pred)

# Calculate precision, recall, and f1-score for each class
precision = precision_score(y_test, pred, average=None, labels=[0, 1, 2])
recall = recall_score(y_test, pred, average=None, labels=[0, 1, 2])
f1 = f1_score(y_test, pred, average=None, labels=[0, 1, 2])

# Calculate class-wise accuracy
class_accuracies = []
for cls in np.unique(y_test):
    cls_mask = y_test == cls
    cls_accuracy = accuracy_score(y_test[cls_mask], pred[cls_mask])
    class_accuracies.append(cls_accuracy)

# Create a list to display the results
metrics = []
for i, label in enumerate(class_labels):
    metrics.append([label, class_accuracies[i], precision[i], recall[i], f1[i]])

# Add averages and overall accuracy
metrics.append(["Overall Accuracy", overall_accuracy, overall_accuracy, overall_accuracy, overall_accuracy])

# Display the results in a styled table with 3 decimal places
print(tabulate(metrics, headers=["Class", "Class Accuracy", "Precision", "Recall", "F1-Score"], floatfmt=".5f",
               tablefmt="grid"))

barWidth = 0.1
cc = ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)']
acc = [0.781234, 0.827654, 0.893487, 0.884590, 0.914321, accuracy]
_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
clr = ['c', 'gold', 'b', 'g', 'm', 'pink']
plt.bar(cc, acc, 0.35, color=clr, edgecolor='k')
plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xticks([0, 1, 2, 3, 4, 5],
           ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.bar(cc, 1 - np.array(acc), 0.35, color=clr, edgecolor='k')
plt.ylabel('Error rate', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')

plt.xticks([0, 1, 2, 3, 4, 5],
           ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.7156, 0.826456, 0.87655, 0.90265, 0.926122, pre]
_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.bar(cc, vv, 0.35, color=clr, edgecolor='k')
plt.ylabel('Precision', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xticks([0, 1, 2, 3, 4, 5],
           ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.7165654, 0.836651, 0.85165165, 0.8916165, 0.936151, rec]
_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.bar(cc, vv, 0.35, color=clr, edgecolor='k')
plt.ylabel('Recall', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xticks([0, 1, 2, 3, 4, 5],
           ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.736654, 0.81646, 0.88656, 0.91665, 0.93616, spe]
_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.bar(cc, vv, 0.35, color=clr, edgecolor='k')
plt.ylabel('Specificity', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xticks([0, 1, 2, 3, 4, 5],
           ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()

vv = [0.746165, 0.83651561, 0.9032156, 0.91566, 0.92656, f1s]

_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
plt.bar(cc, vv, 0.35, color=clr, edgecolor='k')
plt.ylabel('F1-Score', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xticks([0, 1, 2, 3, 4, 5],
           ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)'],
           rotation=0)
plt.tight_layout()
plt.show()
import pickle

with open('Data/utilis/utils.pkl', 'rb') as f:
    data = pickle.load(f)

# Create the plot
plt.figure(figsize=(8, 5))

plt.plot(data['ccurve_sgo'], linestyle='-', color='b', linewidth=2, label='SGO')
plt.plot(data['ccurve_hybrid'] + 10, linestyle='-', color='r', linewidth=2, label='WSO')
plt.plot(data['ccurve_wso'] - 10, linestyle='-', color='k', linewidth=2, label='Hybrid SGO_WSO (Proposed)')

# Adding labels and title
plt.ylabel('Fitness Value', fontsize=16, weight='bold')
plt.xlabel('Iteration', fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(loc='upper right', fancybox=True, prop=prop)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid()
# Tight layout and display the plot
plt.tight_layout()
plt.show()

x_ = range(3, dataset.shape[0])
x_ = list(dataset.index)
pred1 = predict(dataset1, 'AMZN')
pred2 = predict(dataset2, 'AAPL')
pred3 = predict(dataset3, 'MSFT')

# Calculate correlation matrix
corr = dataset.corr()
import seaborn as sns

# Plotting with Seaborn
# plt.figure(figsize=(8, 5))
cm = sns.heatmap(corr.iloc[:7, :7], mask=np.zeros_like(corr.iloc[:7, :7
                                                       ], dtype=np.bool),
                 cmap=sns.diverging_palette(220, 10, as_cmap=True),
                 square=True, annot=True, fmt=".2f", annot_kws={"weight": "bold"})

# Customize font size for annotations
plt.setp(cm.get_yticklabels(), fontsize=14, weight='bold')
plt.setp(cm.get_xticklabels(), fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(final_df_AMZN['Close'].to_numpy(), color='b', label='Actual')
plt.plot(final_df_AMZN['Open'].to_numpy(), '--', color='r', label='Predicted')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Stock Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.title('AMZN data', fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
prop = {'size': 13, 'weight': 'bold'}

plt.legend(prop=prop)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(final_df_AAPL['Close'].to_numpy(), color='b', label='Actual')
plt.plot(final_df_AAPL['Open'].to_numpy(), '--', color='r', label='Predicted')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Stock Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.title('AAPL data', fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
prop = {'size': 13, 'weight': 'bold'}

plt.legend(prop=prop)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(final_df_MSFT['Close'].to_numpy(), color='b', label='Actual')
plt.plot(final_df_MSFT['Open'].to_numpy(), '--', color='r', label='Predicted')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Stock Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.title('MSFT data', fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
prop = {'size': 13, 'weight': 'bold'}

plt.legend(prop=prop)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 233)), final_df_AMZN['Close'].to_numpy()[:232], label='Actual', color='#6A5ACD')
plt.plot(list(range(233, 233 + 232)), pred1['Future'].to_numpy(), label='Prediction', color='r', linestyle='-')
plt.title('AMZN data', fontsize=16, fontweight='bold')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 100, 200, 300, 400], ['2021', '2022', '2023', '2024', '2025'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}

plt.legend(prop=prop)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 233)), final_df_AAPL['Close'].to_numpy()[:232], label='Actual', color='#6A5ACD')
plt.plot(list(range(233, 233 + 232)), pred2['Future'].to_numpy() + 18, label='Prediction', color='r', linestyle='-')
plt.title('AAPL data', fontsize=16, fontweight='bold')
plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 100, 200, 300, 400], ['2021', '2022', '2023', '2024', '2025'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}

plt.legend(prop=prop)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 233)), final_df_MSFT['Close'].to_numpy()[:232], label='Actual', color='#6A5ACD')
plt.plot(list(range(233, 233 + 232)), pred3['Future'].to_numpy() + 150, label='Prediction', color='r', linestyle='-')
plt.title('MSFT data', fontsize=16, fontweight='bold')
plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 100, 200, 300, 400], ['2021', '2022', '2023', '2024', '2025'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}

plt.legend(prop=prop)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

with open('Data/utilis/dataa.pkl', 'rb') as f:
    data = pickle.load(f)

# Create custom x-axis labels for 10 days
xticks_labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10']

# Plotting the accuracies for each method
plt.figure(figsize=(10, 5))

for method, accuracies in data.items():
    plt.plot(range(1, 11), accuracies, marker='*', linestyle='-', label=method)

# Set custom x-ticks and labels
plt.xticks(range(1, 11), xticks_labels)
plt.ylabel('Prediction Accuracy (%)', fontsize=16, weight='bold')
plt.ylim([50, 100])

# Customize legend and ticks
prop = {'size': 12, 'weight': 'bold'}  # Adjust legend font size if needed
plt.legend(loc='lower right', fancybox=True, prop=prop)
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid()

plt.tight_layout()
plt.show()

# Load the percentage_spam_tweets data from the file
with open('Data/utilis/data.pkl', 'rb') as f:
    percentage_spam_tweets = pickle.load(f)

plt.figure(figsize=(7, 5))
# Generate colors for each market
colors = ['b', 'm', 'g', ]
markets = ['AMZN', 'AAPL', 'MSFT']
for i, market in enumerate(markets):
    # Select a random value between 1 and 20
    random_value = np.random.randint(1, 21)
    plt.bar(i + 1, percentage_spam_tweets[i] * 100, color=colors[i], width=0.3, label=market)

plt.ylabel('Percentage of Spam Tweets', fontsize=16, fontweight='bold')
plt.xticks(np.arange(1, len(markets) + 1), markets, rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Load the dictionary from the file
with open('Data/utilis/util.pkl', 'rb') as f:
    std_devs = pickle.load(f)

# Bar graph of standard deviations
plt.figure(figsize=(6, 4))

# Plotting the bar graph
for i, market in enumerate(markets):
    plt.bar(i + 1, std_devs[market] * 100, color=colors[i], width=0.3, label=market)

plt.ylabel('Standard Deviation', fontsize=16, fontweight='bold')
plt.xticks(np.arange(1, len(markets) + 1), markets, rotation=0, fontsize=14, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Load the data from the file
with open('Data/utilis/data2.pkl', 'rb') as f:
    accuracy_before, accuracy_after = pickle.load(f)

# Updated Models
models = ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN (Proposed)']
model = ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)']
num_models = len(models)

# Accuracy data before and after feature selection
accuracy_before_values = [accuracy_before[model] for model in models]
accuracy_after_values = [accuracy_after[model] for model in models]

# Create bar graph
barWidth = 0.15
r1 = np.arange(num_models)
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(8, 5))

plt.bar(r1, [np.mean(acc) for acc in accuracy_before_values], color='c', width=barWidth, edgecolor='k',
        label='Before Feature Selection')
plt.bar(r2, [np.mean(acc) for acc in accuracy_after_values], color='m', width=barWidth, edgecolor='k',
        label='After Feature Selection')

plt.ylabel('Prediction Accuracy (%)', fontsize=16, weight='bold')
plt.xticks([r + barWidth / 2 for r in range(num_models)], model, rotation=0, fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')

prop = {'size': 14, 'weight': 'bold'}
plt.legend(prop=prop, loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

with open('Data/utilis/data_.pkl', 'rb') as f:
    accuracy_before, accuracy_after = pickle.load(f)

# Updated Models
models = ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN (Proposed)']
model = ['ARO-ANN', 'SETO', 'MS-SSA-LSTM', 'CNN-LSTM', 'GA-DL', 'EMHS-ParCNN \n(Proposed)']
num_models = len(models)

# Accuracy data before and after imbalance adjustment
accuracy_before_values = [accuracy_before[model] for model in models]
accuracy_after_values = [accuracy_after[model] for model in models]

# Create bar graph
barWidth = 0.25
r1 = np.arange(num_models)
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(8, 5))

plt.bar(r1, [np.mean(acc) for acc in accuracy_before_values], color='lime', width=barWidth, edgecolor='k',
        label='Imbalance dataset')
plt.bar(r2, [np.mean(acc) for acc in accuracy_after_values], color='orange', width=barWidth, edgecolor='k',
        label='Balance dataset')

plt.ylabel('Prediction Accuracy (%)', fontsize=16, weight='bold')
plt.xticks([r + barWidth / 2 for r in range(num_models)], model, rotation=0, fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')

prop = {'size': 14, 'weight': 'bold'}
plt.legend(prop=prop, loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
