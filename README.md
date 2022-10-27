# org_name_similarity

## Структура проекта и описание задачи
Модуль org_name_similarity - это текстовый парсер, распознающий названия компаний в англоязычном тексте и определяющий их схожесть. Программа принимает на вход текст и возвращает JSON-файл cо словарем, в котором ключи - это названия компаний, а их значения - это списки, состоящие из списков индексов (индекс первой буквы, индекс последней буквы).

### Запуск проекта:

1. Скачайте проект

```bash
git clone https://github.com/dpkaranov/org_name_similarity.git
```
2. Перейдите в директорию проекта

```bash
cd ./org_name_similarity
```
3. Скачайте [модель](https://drive.google.com/u/0/uc?id=1hbftMcPJoL9QGukfBnr3aFb31rqJBAIB&export=download) и распакуйте её в папке ./org_name_similarity/models

4. Положите текстовый файл в папку ./org_name_similarity/texts

5. Запустите скрипт

```bash
python name_sim.py --path <путь/до/файла>
```
или

```bash
python3 name_sim.py --path <путь/до/файла>
```

6. JSON-файл с результатом парсинга появится в папке out

### Возможен и другой вариант использования:

Если есть необходимость определить схожесть каких-либо названий через терминал, можно использовать следующую команду:

```bash
python3 name_sim.py --check <первое название> <второе название>
```

В этом случае программа возвратит булево значение True / False

### Кроме описанных примеров, возможно импортирование класса NameSim в Ваш скрипт.

Убедитесь, что папка с модулем находится в директории Вашего проекта.
В скрипте импортируйте класс.

```python
from org_name_similarity.name_sim import NamSim

namesim = NamSim()

print(namesim.check_similarity('Saudi Aramco', 'Rosneft'))
```

### Структура проекта:
```shell
├── __init__.py
├── name_sim.py
├── data
      └── data.csv
├── images
├── metrics
      └── comp_with_128.csv
      └── first_metrics.csv
      └── hard_comp_with_128.csv
├── models
├── notebooks
      └── 1 Создание мультиязычной базы данных.ipynb
      └── 2 Загрузка моделей трансформеров.ipynb
      └── 3 Обучение трансформеров на собственном датасете.ipynb
      └── 4 Сравнение обученного трансформера с необученными моделями.ipynb
      └── 5 Подготовка нового датасета.ipynb
      └── 6 Обучение новой модели трансформера на новом датасете.ipynb
      └── 7 Создание модели с аутпутом 128 и сравнение моделей на тестовых данных.ipynb
├── texts
├── out
```

## Метод решения и его обоснование

Для решения задачи определения схожести названий проведено дообучение (fine-tuning) модели-трансформер с помощью фреймворка sentence-transformes. Трансформеры буквально "захватили" NLP, демонстрируя высокие показатели по сравнению с RNN, LSTM и др. технологиями машинного обучения. Их привлекательность для обработки естественного языка заключается в применении механизма внутреннего внимания, устанавливающего свзязи между отдельными словами.

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/attention.png?raw=true)

В качестве альтернативы рассматривались tensorflow, torch, spacy, но в итоге выбор был сделан в пользу sentence-transformes. Преимущества ST: высокопроизводительные модели, автоматизированная настройка гиперпараметров и простота использования.
Технология достаточно проста: списки строк (сначала первые названия компаний, потом вторые) отправляются в модель, которая возвращает эмбеддинги. Полученные эмбеддинги сравниваются с помощью измерения косинусного расстояния между ними. Если косинусное расстояние близко к 1, то названия организаций схожи, если к 0, то нет.
Помимо этого для поиска имен в тексте (задача NER) используется Spacy.

## Набор данных (датасет)

Для обучения модели использовался датасет, состоящий из 497814 строк и 4 столбцов - номер пары (pair_id), первое название (name_1), второе название (name_2) и отметка схожести (is_duplicated). При этом 99 % пары были отмечены как непохожие (0), и менее 1 % - похожие (1). К тому же, названия компаний включали в себя разный шум (символы, лишние буквы из других алфавитов и т.д.). Соответственно, необходимо было тщательно подготовить датасет перед обучением.

## Этапы решения задачи

1. Подготовка данных к обучению модели;

2. Обучение моделей и сравнение их показателей;

3. Тестирование модели и выбор лучшей;

4. Создание скрипта;

## Эксперименты

### Эксперимент №1 Обучение модели на мультиязычном датасете

_Гипотеза: перевод всех схожих пар названий организаций и такого же количества непохожих пар на разные языки позволит увеличить датасет в несколько раз, а классы будут представлены в нем в одинаковых пропорциях._

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/diagram1.png?raw=true)

Перевод названий организаций осуществлен с помощью библиотеки transliterate. В качестве модели выбрана MiniLM-L12-v2.

Реализация этого эксперимента представлена в папке notebooks, в файлах 1 - 4.

В 4 ноутбуке проведено сравнение обученной модели с необученными.

_Сравнение по метрикам обученной модели MiniLM-L12-v2 с её необученным аналогом и остальными трансформерами_

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/acc.png?raw=true)

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/recall.png?raw=true)

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/f1.png?raw=true)

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/fbeta.png?raw=true)

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/ham.png?raw=true)

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/jac.png?raw=true)

_Таблица №1 Сводная таблица_

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/table1.png?raw=true)

_Вывод: несмотря на перевод слов на разные языки и мультиязычную базу самой модели, модель переобучилась. Это не столько очевидно по результатам тестов на тестовом наборе данных, выделенном из общего датасета, сколько по "ручной проверке". Так, например, модель не смогла отличить Газпром от Роснефти, а с этим примером может справиться даже расстояние Левенштейна. Тем не менее, этот эксперимент позволил отбросить одну из гипотез, а сравнение разных моделей показало, что гораздо больше смысла в обучении distiluse-base-multilingual-cased-v2 (на базе BERT), которая даже без обучения показала очень высокие показатели._

### Эксперимент №2 Создание нового датасета и обучение distiluse-base-multilingual-cased-v2

В этом эксперименте использовался иной принцип подготовки данных:

1. Найдены все уникальные значения (порядка 17 тыс. на оба столбца);

2. Удалены лишние символы (т.е. все буквы не относящиеся к латинскому алфавиту), пустые и короткие строки;

3. Все длинные названия разделены по пробелу, вторые слова перемешаны и добавленны к первым. Также обозначены часто встречающиеся слова, заканчивающие названия англоязычных организаций, которые также перемешаны и добавлены к основным названиям.

_Часто встречающиеся слова_

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/suf.png?raw=true)

Подробнее с ходом эксперимента можно ознакомиться в блокнотах №6 и №7.

### Эксперимент №3 Уменьшение размерности output-слоя при сохранении точности

В блокноте №7 представлен эксперимент по созданию модели с уменьшенной размерностью последнего слоя (с 512 до 128). Может возникнуть вопрос почему сразу, еще до обучения модели не внести коррективы? В таком случае мы могли потерять важные фичи, кроме того, этот эксперимент предполагает изучение возможности снижения размера исходящего эмбеддинга для уже обученных моделей. Как видно из представленных данных, модель не отстает по точности от своего аналога.

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/table2.png?raw=true)

## Оценка производительности

На тестовых данных distiluse-base-multilingual-cased-v2 показала невероятные 1.0 по всем метрикам. Проверка модели на сложных примерах может показать её реальное качество. Почему сложные? Потому что часть примеров модель никогда не видела (например, на кириллице или российских наименований с сокращенным указанием формы собственности в начале строки "ООО", "ОАО" и т.д. ). Примеров немного, но даже на них модель distiluse-base-multilingual-cased-v2 продемонстрировала хорошие результаты, отличив Bridge от Бриджит Бардо, Зенит от Спартака и т.д. Модель не справилась с Роскачеством и Россетями, что показаывает о необходимости подготовки русскоязычной модели отдельно.

_Проверка модели на сложных примерах_

![alt text](https://github.com/dpkaranov/org_name_similarity/blob/master/images/hard2.png?raw=true)

В папке metrics хранятся сводные таблицы.
