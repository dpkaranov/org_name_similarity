# org_name_similarity

## Структура проекта и описание задачи
Модуль org_name_similarity - это текстовый парсер, распознающий названия компаний в англоязычном тексте и определяющий их схожесть. Программа принимает на вход текст и возвращает json-file cо словарем, в котором названия компаний - ключи, а значения - списки, состоящие из кортежей с индексами.

### Запуск проекта:

1. Скачайте проект

```
git clone https://github.com/dpkaranov/org_name_similarity.git
```
2. Перейдите в директорию проекта

```
cd ./org_name_similarity
```
3. Скачайте модель и распакуйте её в папке ./org_name_similarity/models

```
wget https://drive.google.com/file/d/1hbftMcPJoL9QGukfBnr3aFb31rqJBAIB/view?usp=sharing
```
4. Положите текстовый файл в папку ./org_name_similarity/texts

5. Запустите скрипт

```
python name_sim.py <путь/до/файла>
```
или

```
python3 name_sim.py <путь/до/файла>
```

### Возможен и другой вариант использования:

Если есть необходимость определить схожесть каких-либо названий через терминал, можно использовать следующую команду:

```
python3 name_sim.py --check <первое название> <второе название>
```

В этом случае программа возвратит булево значение True / False

### Кроме описанных примеров, возможно импортирование класса NameSim в Ваш скрипт.

Убедитесь, что папка с модулем находится в директории Вашего проекта.
В скрипте импортируйте класс.

```
from org_name_similarity import NameSim

namesim = NameSim()

print(namesim.check_similarity('Saudi Aramco', 'Rosneft'))
```

### Структура проекта:

> org_name_similarity
> > data
> > images
> > metrics
> > models
> > notebooks
> > > 1 Создание мультиязычной базы данных.ipynb
> > > 2 Загрузка моделей трансформеров.ipynb
> > > 3 Обучение трансформеров на собственном датасете.ipynb
> > > 4 Сравнение обученного трансформера с необученными моделями.ipynb
> > > 5 Подготовка нового датасета.ipynb
> > > 6 Обучение новой модели трансформера на новом датасете.ipynb
> > > 7 Создание модели с аутпутом 128 и сравнение моделей на тестовых данных.ipynb


## Метод решения и его обоснование

Для решения задачи проведено дообучение (fine-tuning) модели-трансформер с помощью фреймворка sentence-transformes. Трансформеры буквально "захватили" NLP, демонстрируя высокие показатели по сравнению с RNN, LSTM и др. технологиями. Их привлекательность для обработки естественного языка заключается в применении механизма внутреннего внимания, устанавливающего свзязи между отдельными словами. В качестве альтернативы рассматривались tensorflow, torch, spacy, но в итоге выбор был сделан в пользу sentence-transformes. Преимущества ST: высокопроизводительные модели и простота использования.

## Этапы решения задачи

## Эксперименты

### Эксперимент №1

### Эксперимент №2

### Эксперимент №3

## Оценка производительности
