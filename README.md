# Matching товаров ООО "ПРОСЕПТ"

## Введение

**ООО «ПРОСЕПТ»** — российская производственная компания, специализирующаяся
на выпуске профессиональной химии. В своей работе используют опыт ведущих
мировых производителей и сырье крупнейших химических концернов. Производство и
логистический центр расположены в непосредственной близости от Санкт-Петербурга,
откуда продукция компании поставляется во все регионы России.
Сайт: https://prosept.ru/


**Введение в задачу**:

Заказчик производит несколько сотен различных товаров бытовой и промышленной
химии, а затем продаёт эти товары через дилеров. Дилеры, в свою очередь,
занимаются розничной продажей товаров в крупных сетях магазинов и на онлайн
площадках.

Для оценки ситуации, управления ценами и бизнесом в целом, заказчик
периодически собирает информацию о том, как дилеры продают их товар. Для этого
они парсят сайты дилеров, а затем сопоставляют товары и цены.
Зачастую описание товаров на сайтах дилеров отличаются от того описания, что даёт
заказчик. Например, могут добавляться новый слова (“универсальный”,
“эффективный”), объём (0.6 л -> 600 мл). Поэтому сопоставление товаров дилеров с
товарами производителя делается вручную.
Цель этого проекта - разработка решения, которое отчасти автоматизирует процесс
сопоставления товаров. Основная идея - предлагать несколько товаров заказчика,
которые с наибольшей вероятностью соответствуют размечаемому товару дилера.
Предлагается реализовать это решение, как онлайн сервис, открываемый в веб-
браузере. Выбор наиболее вероятных подсказок делается методами машинного
обучения.

**Документация к предоставленным данным**:

Заказчик предоставил несколько таблиц (дамп БД), содержащих необходимые
данные:

1 marketing_dealer - список дилеров;

2 marketing_dealerprice - результат работы парсера площадок дилеров:

- product_key - уникальный номер позиции;

- price - цена;

- product_url - адрес страницы, откуда собраны данные;

- product_name - заголовок продаваемого товара;

- date - дата получения информации;

- dealer_id - идентификатор дилера (внешний ключ к marketing_dealer)


3 marketing_product - список товаров, которые производит и распространяет
заказчик;

- article - артикул товара;

- ean_13 - код товара (см. EAN 13)

- name - название товара;

- cost - стоимость;

- min_recommended_price - рекомендованная минимальная цена;

- recommended_price - рекомендованная цена;

- category_id - категория товара;

- ozon_name - названиет товара на Озоне;

- name_1c - название товара в 1C;

- wb_name - название товара на Wildberries;

- ozon_article - описание для Озон;

- wb_article - артикул для Wildberries;

- ym_article - артикул для Яндекс.Маркета;

4 marketing_productdealerkey - таблица матчинга товаров заказчика и товаров
дилеров

- key - внешний ключ к marketing_dealerprice

- product_id - внешний ключ к marketing_product

- dealer_id - внешний ключ к marketing_dealer

## План работ

**До дедлайна 19:00 29 ноября**

1. Команда знакомится с предоставленными данными.

2. Формулируется DS задача, утверждается единые схема валидации решений и метрика качества.

3. Выбирается основной способ решения задачи (модель первого этапа с функционалом финального решения),
    который будет представлен к дедлайну 29 ноября:
    - Рассматриваются и валидируются разные способы предобработки входных данных модели.
    - Рассматриваются и валидируются разные ml движки решения.


4. Подготовка модели этапа к сдаче на ревью в том виде, в котором ею сможет пользоваться BackEnd департамент команды.

5. Подготовка репозитория решения с jupyter notebook, содержащим основные вехи разработки модели первого этапа.

**После дедлайна DS до единого дедлайна**

6. Тюнинг схемы предобработки неймингов, валидация результатов.

7. Дообучение берт-модели.

**8**. Оформление документации и ожидание результатов хакатона.

## Формулирование DS задачи

**Роль алгоритма в функционале приложения**

Наш алгоритм должен помочь разметчику сопоставить товар диллера с одним из нескольких сотен товаров фирмы заказчика. Важно отметить, что финальное решение принимает именно разметчик. Так вот, насколько видит наша команда, алгоритм для каждого из предложенных товаров диллеров должен вернуть ранжированный список всех товаров заказчика так, чтобы релевантный айтем оказался максимально высоко в топе. Таким образом перед нами тривиальная задача ранжирования.

#### Особенности задачи

- Для товара диллера у заказчика есть только один релевантный айтем.
- Основными признаками для матчинга выступают нейминги товаров.
- Товары диллеров всегда новые, а множество товаров заказчика меняется редко.

#### Метрика качества

В качестве метрики качества мы утвердили Среднеобратный ранг (Mean Reciprocal Rank): $MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$ ,
где ${\text{rank}_i}$ это позиция релевантного айтема заказчика в ранжированном списке а ${N}$ это мощность множества товаров дилеров. Метрика выбрана по ряду причин, и по нашему общему мнению идеально совпадает с нашей задачей. MRR это ничто иное как средняя позиция правильного ответа (совпадения / релевантного item), для удобства шкалированная от 0 до 1. Чтобы из MRR получить среднюю позицию правильного ответа, достаточно делить на неё единицу. В случае одного релевантного товара nDCG (популярная метрика ранжирования) и MRR будут равны с точностью до константы.

#### Схема валидации

Т.к. соответствия товаров диллеров товарам заказчика не изменяются со временем, обычная кросс-валидация идеально нам подходит. Будем валидироваться на 5 фолдах, в тестовой выборке каждого фолда будет примерно 300-400 соответствий.

#### Признаки айтемов 

Основным признаком для сопоставления являются нейминги товаров. Таким образом в качестве решения мы представим функцию, помещающую все нейминги товаров заказчика в единое векторное пространство с помощью текстового векторизатора. Для формирования списка рекомендаций нейминг товара дилера помещается в то же пространство, а затем товары заказчика ранжируются по косинусной близости. В итоге уже обученная модель будет принимать массив с неймингами товаров дилера, а возвращать двумерный массив с рекомендациями.

## Результаты

В ходе работ нам удалось достичь **0.91 MRR**, это означает что среднее значение позиции правильного ответа приблизительно равно **1.095**. Вероятность того, что правильный ответ окажется **на первом месте** рекомендаций приблизительно равно **87%**, **95%** что войдёт в **топ 3**, **98%** - в **топ 10**. Для этого мы дообучали на известных взаимодействиях берт-дистиллят https://huggingface.co/cointegrated/rubert-tiny2 . Дообученный bert-tiny2 сохранен в репозитории https://huggingface.co/micoff/rubert-tiny2-tuned-for-prosept . 

## Инструкция к применению:

```
from main import *


model = DistanceRecommender(
    vectorizer=TransformerVectorizer(),
    simularity_func=cosine_similarity,
    text_prep_func=string_filter_emb
)

model.from_pretrained()

names = ['Герметик акриловый  цвет белый , ф/п 600 мл. (12 штук )',
         'Гель эконом-класса для мытья  посуды вручную. С ароматом яблокаCooky Apple Eконцентрированное средство / 5 л ПЭТ',
         'Средство для удаления ржавчины и минеральных отложений щадящего действияBath Acid  концентрат 1:200-1:500 / 0,75 л ',
         'Антисептик многофункциональный ФБС, ГОСТ / 5 л',
         'Гелеобразное средство усиленного действия для удаления ржавчины и минеральных отложенийBath Extraконцентрат 1:10-1:100 / 0,75 л']


print(model.recommend(names))
```

## Обращение к модели с помощью API

С моделью можно взаимодействовать с помощью API реализованного на FastApi. 


1. Необходимо запустить сервер uvicorn локально. \n 
*Так же необходимо выставить порт, если запускать проект вместе с backend'ом локально **--port 8001***

```
uvicorn api.api:app 
```

API имеет один эндпоинт - *'/machine-matching/'* и принимает POST-запрос с телом запроса:
```
{
    "name_dealer_product": "наименование продукта дилера"
}
```

И отдаёт ранжированный список id's товаров заказчика:
```
[446 464 466 ... 115 482 436]
```


## Необходимое окружение для работы модели:

```
pandas==2.1.3
transformers==4.35.2
torch==2.1.1
scikit-learn==1.3.2
fastapi==0.104.1
uvicorn==0.24.0.post1
num2words==0.5.13
sentence-transformers==2.2.2
```
