# deep-learning-Standford-dogs-classification

#run code(.ipynb) at Google Colab

#Standford Dogs Dataset  (training 12000  test 8580   120 species)

#嘗試了很多model  最後選擇用 InceptionV3 當base model 再加上 

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dropout(0.3))

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(120, activation = 'softmax'))


#relu 可以加快運算  softmax則是在處理多類別分類時用的

#注意事項:

資料增益的地方有新增一些可以設定的參數

在 testBatch = testDataGenerator.flow_from_directory  之中要加一行 shuffle=False,

因為他預設是True 而test 不需要進行shuffle

learning rate 不能條太高 這邊設定為0.001 batch_size=32 epoch=30   image_size調整到(299,299) 因為要大一點才能比較好的擷取特徵



