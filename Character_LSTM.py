'''
	There is a problem that you've to fix now.
	There should be a length but if the word in short then 
	the neural network shouldn't crash.

	This shit is not necessary
'''

import numpy as np 
import keras
from Models import example_model
from keras.models import load_model
np.set_printoptions(threshold=np.nan) # to see the entire numpy data for debugging 

text_file = open("test.txt", encoding='utf8').read()

each_data = list(sorted(set(text_file)))

#get the max word

listed_data = text_file.split()
# print(listed_data)

length = []

for each_listed_data in listed_data:
	length.append(len(each_listed_data))

max_length = max(length) #14 # no need to check!

#ends here


int_to_chars = dict((char,num) for (char,num) in enumerate(each_data))
chars_to_int = dict((num,char) for (char,num) in enumerate(each_data))

print("All Characters to Integers")
print(chars_to_int)
print()
data_val = []

for each_word in text_file:
	chars_value = chars_to_int[str(each_word)]
	data_val.append(chars_value)

# print(data_val)

# Time to write the algorithm

A = 0
B = 5
data_X = []
data_Y = []

for i in range(len(data_val)):
	try:
		X_int_data = data_val[A:B]
		Y_int_data = data_val[A+5]
		data_X.append(X_int_data)
		data_Y.append(Y_int_data)
		A +=1
		B +=1
	except:
		pass

# print(data_X)
# print()
# print(data_Y)

# ends here

X_data = np.array(data_X)
Y_data = np.array(data_Y)

X_data = X_data.reshape(X_data.shape[0],1,5)
# X_data = X_data /255
# print(Y_data.shape[0])
num_classes = len(each_data)

Y_data = keras.utils.to_categorical(Y_data, num_classes)
# print(Y_data.shape)
# print(Y_data)

def prediction_stage():# try to make any evaluation if you can 
	input_sentence = input("Enter a 6 charactered sentence:")
	model_name= "CLevel_w.h5"
	required_length = 5

	input_sent_list = list(input_sentence)
	# print(input_sent_list)

	# Fixing the length of the input text for the model

	text_length = len(input_sentence)

	if text_length > required_length:
		text_array = []
		for i in range(required_length):
			text_array.append(input_sent_list[i])

		input_sent_list = text_array
	# ends here
	print()
	print("Data to be fed = ", input_sent_list)
	print()

	loaded_model = load_model(model_name)
	opt = keras.optimizers.Adam(lr=0.001)

	loaded_model.compile(
			loss='categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
		)
	
	data_val = []

	for each in  input_sent_list:
		chars_val = chars_to_int[str(each)]
		data_val.append(chars_val)

	# print(data_val)
	data_val = np.array([data_val])
	data_val = data_val.reshape(1,data_val.shape[0], data_val.shape[1])
	# print(data_val.shape)

	predicted_data_val = (loaded_model.predict_classes(data_val))
	print(predicted_data_val)
	print(int_to_chars[int(predicted_data_val)])


def Continue_Training(nb_epochs): # I don't know whether this is the right code
	loaded_model = load_model("CLevel_w.h5")
	opt = keras.optimizers.Adam(lr=0.001)
	loaded_model.compile(
			loss='categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
		)

	loaded_model.fit(X_data, Y_data, epochs=nb_epochs, batch_size=132)
	loaded_model.save("CLevel_w_continuation.h5")


def Text_generation():
	# model_name = "CLevel_w.h5"
	model_name = "CLevel_w_continuation.h5"
	loaded_model = load_model(model_name)
	opt = keras.optimizers.Adam(lr=0.001)

	loaded_model.compile(
			loss='categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
		)

	input_sentence = input("Enter a sentence:")
	required_length = 5
	input_s_list = list(input_sentence)

	# Fixing the length of the input text for the model

	text_length = len(input_sentence)

	if text_length > required_length:
		text_array = []
		for i in range(required_length):
			text_array.append(input_s_list[i])
		text_array = ''.join(text_array)

		input_sentence = text_array
	else:
		# pass
		# The word input in list => input_s_list
		# pass # add a space in the end when the word in shorter than 5
		if text_length < required_length:
			text_array = []

			for i in range(required_length):
				try:
					text_array.append(input_s_list[i])
				except:
					text_array.append(" ")

			text_array = ''.join(text_array)
			input_sentence = text_array

	input_s = input_sentence

	# ends here
	X = 1
	Y = 6
		
	final_data = []

	# i suppose the loop should start from here
	for i in range(100):
		def Letter_predict(data_val):
			predicted_data_val = (loaded_model.predict_classes(data_val))
			# predicted_data_str = (int_to_chars[int(predicted_data_val)])
			# print(predicted_data_str)
			return (int_to_chars[int(predicted_data_val)])
		# print()
		print("Data to be fed = ", list(input_sentence))
		# print()
		
		data_val = []

		for each in input_sentence:
			chars_val = chars_to_int[str(each)]
			data_val.append(chars_val)

		# print(data_val)
		data_val = np.array([data_val])
		data_val = data_val.reshape(1,data_val.shape[0], data_val.shape[1])
		# print(data_val.shape)

		predicted_data_str = Letter_predict(data_val)
		final_data.append(predicted_data_str)
		# print(predicted_data_str)
		new_sentence = input_sentence + str(predicted_data_str)
		# print("add new sentence",new_sentence)

		# array_list_sent = list(sentence)
		input_sentence = new_sentence[X:Y]
		# print(X,Y)
		# print("input new sentence", input_sentence)
	text = ''.join(final_data)
	print(input_s + text)

# example_model(X_data,Y_data,3000)
# prediction_stage()
Text_generation()
# Continue_Training(3000)
