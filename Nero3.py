import numpy as np

def activarion_function(x):
    return 1 / (1 + np.exp(-x))
def proizwod(out):
    return (1 - out) * out

# Инициализация переменных
o_ideal = 1 # Идеальный результат
nero_input = [1, 0] # Входные данные: значения a и b
synaps_in_h1 = [0.45, -0.12] # Входные веса синапсов в первый скрытый нейрон
synaps_in_h2 = [0.78, 0.13] # Входные веса синапсов во второй скрытый нейрон
h1_nero = h2_nero = 0.0 # Значения самих скрытых нейронов
synaps_in_o = [1.5, -2.3] # Входящие синапсы в выходной нейрон
o_nero = float # Результат работы нейросети
lerning_speed = 0.7 # скорость обучения
moment = 0.3 # момент

# Первая итерация вычислений

print("Первый пример задачи: a = 0, b = 1, значит c = 1")
print("Второй пример задачи: a = 0, b = 0, значит c = 0")
print("Третий пример задачи: a = 1, b = 1, значит c = 0")
print()
print("Условие: a = 1, b = 0, c = ?")
print()

for i in range(len(synaps_in_h1)): # результат для первого скрытого нейрона
    h1_nero += (nero_input[i] * synaps_in_h1[i])
h1_nero = activarion_function(h1_nero) # преобразование результата в стандартный вид
for i in range(len(synaps_in_h2)): # Для второго скрытого нейрона
    h2_nero += (nero_input[i] * synaps_in_h2[i])
h2_nero = activarion_function(h2_nero)

o_nero = h1_nero * synaps_in_o[0] + h2_nero * synaps_in_o[1]
o_nero = activarion_function(o_nero) # подсчет значения вывода

error = ((o_ideal - o_nero) ** 2) / 1 # вероятность ошибки

print("Начальный результат: ", o_nero, ". Oшибка: ", error, "%", sep='')

for k in range(5): # количество эпох
    for i in range(10): # тренировочный сет, МОР и обновление весов
        grad = [0, 0, 0, 0, 0, 0] # массив для градиентов, 0 - последний вес. 5 - первый
        delta_synaps = [0, 0, 0, 0, 0, 0] # изменение весов синапсов
        delta_synaps_back = [0, 0, 0, 0, 0, 0] # изменение предыдущих весов синапсов

        delta_o = (o_ideal - o_nero) * proizwod(o_nero)
        o_nero += delta_o # подсчет вывода

        # Работа с H2
        delta_h2 = proizwod(h2_nero) * (synaps_in_o[0] * delta_o)
        grad[0] = h2_nero * delta_o # градиент для шестого веса
        if i == 0: # при первой итерации delta_synaps_back = 0
            delta_synaps[0] = lerning_speed * grad[0]
            delta_synaps_back[0] = delta_synaps[0]
        delta_synaps[0] = lerning_speed * grad[0] + moment * delta_synaps_back[0] # Вычисление изменения веса синапса
        synaps_in_o[1] += delta_synaps[0] # Обновление веса
        delta_synaps_back[0] = delta_synaps[0] # Запись предыдущего изменения синапса для использования в формуле

        # Аналогично с H1
        delta_h1 = proizwod(h1_nero) * (synaps_in_o[0] * delta_o)
        grad[1] = h1_nero * delta_o
        if i == 0:
            delta_synaps[1] = lerning_speed * grad[1]
            delta_synaps_back[1] = delta_synaps[1]
        delta_synaps[1] = lerning_speed * grad[1] + moment * delta_synaps_back[1]
        synaps_in_o[0] += delta_synaps[1]
        delta_synaps_back[1] = delta_synaps[1]

        # Работа с I2
        grad[2] = nero_input[1] * delta_h1
        grad[3] = nero_input[1] * delta_h2
        if i == 0:
            delta_synaps[2] = lerning_speed * grad[2]
            delta_synaps_back[2] = delta_synaps[2]

            delta_synaps[3] = lerning_speed * grad[3]
            delta_synaps_back[3] = delta_synaps[3]

        delta_synaps[2] = lerning_speed * grad[2] + moment * delta_synaps_back[2]
        synaps_in_h1[1] += delta_synaps[2]
        delta_synaps_back[2] = delta_synaps[2]

        delta_synaps[3] = lerning_speed * grad[3] + delta_synaps_back[3]
        synaps_in_h2[1] += delta_synaps[3]
        delta_synaps_back[3] = delta_synaps[3]

        # Работа с I1
        grad[4] = nero_input[0] * delta_h1
        grad[5] = nero_input[0] * delta_h2
        if i == 0:
            delta_synaps[4] = lerning_speed * grad[4]
            delta_synaps_back[4] = delta_synaps[4]

            delta_synaps[5] = lerning_speed * grad[5]
            delta_synaps_back[5] = delta_synaps[5]

        delta_synaps[4] = lerning_speed * grad[4] + moment * delta_synaps_back[4]
        synaps_in_h1[0] += delta_synaps[4]
        delta_synaps_back[4] = delta_synaps[4]

        delta_synaps[5] = lerning_speed * grad[5] + delta_synaps_back[5]
        synaps_in_h2[0] += delta_synaps[5]
        delta_synaps_back[5] = delta_synaps[5]

        error = ((o_ideal - o_nero) ** 2) / 1
    print("Результат после ", k + 1, " эпох: ", o_nero, ". Oшибка: ", error, "%", sep='')

output = round(o_nero, 0)
error = round(error, 3)
print("Конечный результат после обучения: c = ", output, ". Oшибка: ", error, "%", sep='')
