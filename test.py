from main import *


model = DistanceRecommender(
    vectorizer=InfloatVectorizer(),
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
