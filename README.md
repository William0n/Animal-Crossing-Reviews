# Animal Crossing: New Reviews 
## Introduction 

With the release of Nintendo's new 2020 Animal Crossing game, it is inevitable that there will also be a variety of reviews on the horizon. This was a great opportunity to apply some Natural Language Processing (NLP) while also seeing how good the game is before I purchase my own copy. In this project, my goal is to utilize Tensorflow's text preprocessing and encoding functions on the reviews and predict whether or not the review is positive or negative. 


## Packages and Resources  
**Packages:**
  - Pandas
  - Numpy
  - sklearn
  - Tensorflow/keras
  - Matplotlib
  - Collections 


The reviews were found on Metacritic. Raw data can also be viewed [here](https://github.com/William0n/Animal-Crossing-Reviews/blob/master/AC%20data.csv)

## Data Preprocessing
The data set contained over 2000 different reviews that weren't exactly the cleanest, as such, the following changes were applied to the data:
  - Reviewer's usernames and dates removed
  - Changed `grade` variable to only include 2 types of reviews (i.e. the review is 1 for positive reviews and 0 for negative reviews)
  - Replaced any unnecessary expressions and symbols from reviews with empty spaces
  - Padding added for any reviews less 400 characters in length

Text Encoding

Following the replacement of expressions and symbols from the texts, a word dictionary was created to keep track of the 2000 most commonly used words in the reviews. Moving on, the text data was then encoded using Tensorflow's `lookup.StaticVocabularyTable` function. This function uses the previously mentioned dictionary to uniquely assign an ID number to replace the words in the review (the ID number is based on where the word is in the word dictionary). If a word is not found in the dictionary, it will be added to an out of vocabulary (OOV) bucket and assigned a new unique ID number. Given the number of reviews in this data set, it would be unlikely to find too many words that are not already found in the dictionary, therefore, the bucket size was set to 500. 

## Model

In terms of modeling, a sequential model was used with a combination of Gated Reccurent Unit (GRU) and normal dense layers. However, what was different for this project was the use of a bidirectional layer. A normal reccurent network only reads current and past data, however, a bidirectional layer will allow the model to read data that is future. As a result, this type of layer was chosen with the intent of increasing the model's accuracy when training. The model can be seen below:

```
embed_size = 150
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size + oov_bucket, embed_size, input_shape = [None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences = True)),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
 ```

## Results 

|                | Training      | Validation   | Test        |
| -------------  | ------------- | -------------|-------------|
| Accuracy       | 0.998         | 0.838        | 0.818       |
| Loss           | 0.009         | 0.869        | 0.994       |


<img src = "imgs/training results.png">

## Conclusion 

Overall, the model performed extremely well on the training set with an accuracy of 99.8%. Although there was a noticeable discrepancy between the training set and the validation set (~15%), the model's performance on the validation set was still relatively high as the accuracy was almost at 85%. In addition, the model achieved an accuracy of 82% on the test set which was quite close to the validation set results. It should also be mentioned that the model only ran for a total of 10 epochs, thus, the results could potentially be significantly improved if the number of epochs increased.


















