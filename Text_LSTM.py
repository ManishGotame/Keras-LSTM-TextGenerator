import numpy as np 
np.set_printoptions(threshold=np.nan) # to see the entire numpy data for debugging 
import keras
from keras.models import load_model

from Models import example_model

model_name = "Example_code_weight_2.h5" # this is the model's weight

text_file = open("test.txt", encoding="utf8").read()
text_file = text_file.lower()
text_file = text_file.split()
# print(text_file)

text_data = sorted(set(text_file))
dictionary = dict((char, num) for (num,char) in enumerate(text_data))
reversed_dictionary = dict((num,char) for (num,char) in enumerate(text_data))

# print(dictionary)
X_data_file = []

for each_word in text_file:
	char_to_int = dictionary[each_word]
	X_data_file.append(char_to_int)

X_data = []
Y_data = []


A = 0 
B = 3
for i in range(len(X_data_file) - 3):
	try:
		X_data_ = X_data_file[A:B]
		num = A+3
		Y_data_ = X_data_file[num]
		A = A + 1 
		B = B + 1

		X_data.append(X_data_)
		Y_data.append(Y_data_)
	except Exception as E:
		pass

X_data = np.array(X_data)
Y_data = np.array(Y_data)

X_data = X_data.reshape(X_data.shape[0],1,3) # fix this please # not needed

# print(X_data[0])

datas_Y = keras.utils.to_categorical(Y_data)
print(datas_Y.shape)

def predict(reversed_dictionary, dictionary): # a single word prediction
	
	'''
		This is only for a single word prediction, not for text generation.
		Input 3 word => ['which', 'way','she']
		prediction => ['knew']

	'''
	opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
	loaded_model = load_model(model_name)
	loaded_model.compile(
		    loss='categorical_crossentropy',
		    optimizer=opt,
		    metrics=['accuracy']
		   )

	input_data = input("Enter 3 words:")

	input_data = input_data.lower().split()
	# print(input_data)
	word_matrix = []
	for sep_word in input_data:
		data = dictionary[sep_word] # this lil shit is limited right now
		word_matrix.append(data)

	num_matrix = np.array([word_matrix])
	# word_matrix = word_matrix.T
	# print(word_matrix.shape)
	data = num_matrix.reshape(1,1,3)

	prediction = (loaded_model.predict_classes(data))
	print("class:", prediction)
	print(reversed_dictionary[int(prediction)])
	# print(input_data + reversed_dictionary[int(prediction)])



def Text_Generator(reversed_dictionary, dictionary): # continous text prediction
	
	'''
		This is the text Generator. 
		input => ['as','she','coudn't]
		text => ['words','words','words']
	'''

	opt = keras.optimizers.Adam(lr=0.001)
	loaded_model = load_model(model_name)
	loaded_model.compile(
		    loss='categorical_crossentropy',
		    optimizer=opt,
		    metrics=['accuracy']
		   )

	# workout the input here

	def prediction_model(num_matrix, loaded_model, reversed_dictionary):
		prediction = (loaded_model.predict_classes(num_matrix))
		predicted_word = reversed_dictionary[int(prediction)]
		return str(predicted_word)



	text_data = input("Enter the initialization 3 words: ")
	X = 0
	Y = 3

	final_output = []

	for i in range(100):
		init_data = text_data.lower().split()
		init_data = init_data[X:Y]

		# remove it to know what is getting fed into the prediction model
		# print("Data to be fed", init_data) 
		
		num_matrix = []
		for sep_word in init_data:
			data = dictionary[sep_word]
			num_matrix.append(data)

		num_matrix = np.array([num_matrix])
		num_matrix = num_matrix.reshape(1,1,3)

		output = prediction_model(num_matrix, loaded_model, reversed_dictionary)
		final_output.append(output)
		text_data = (str(text_data) + " " + output)
		X += 1
		Y +=1
		# print(text_data)

	text = ' '.join(final_output)
	print(text)



if __name__ == '__main__':
	# example_model(X_data,datas_Y) # use this for training

	# predict(reversed_dictionary,dictionary) #single word prediction
	Text_Generator(reversed_dictionary, dictionary) #Text_generation
