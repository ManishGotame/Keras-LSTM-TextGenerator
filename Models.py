def model_1():
	from keras.layers import Dense, LSTM, Dropout
	from keras.models import Sequential
	
	model = Sequential()

	# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
	model.add(LSTM(128, input_shape=(1,3), activation='relu', return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(128, activation='relu'))
	model.add(Dropout(0.1))

	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(datas_Y.shape[1], activation='softmax'))

	opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

	# Compile model
	model.compile(
	    loss='categorical_crossentropy',
	    optimizer=opt,
	    metrics=['accuracy']
	)

	model.fit(X_data, datas_Y, epochs=1000)
	model.save("model_weights.h5")



def model_2(X_data,datas_Y):
	from keras.layers import Dense, LSTM, Dropout
	from keras.models import Sequential
	from keras.optimizers import RMSprop
	
	model = Sequential()

	model.add(LSTM(128, input_shape=(1,3), activation='relu', return_sequences=True))
	model.add(Dropout(0.250))

	model.add(LSTM(256, activation='relu', return_sequences=True))
	model.add(Dropout(0.250))

	model.add(LSTM(512, activation='relu', return_sequences=False))
	model.add(Dropout(0.250))

	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.250))

	# model.add(Dense(datas_Y.shape[1], activation='softmax'))
	model.add(Dense(428, activation='softmax'))

	# Compile model
	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.fit(X_data, datas_Y, epochs=1000, batch_size=1)
	model.save("model_weights.h5")


# the reliable one
def example_model(X_data, datas_Y):
	from keras.layers import Dense, LSTM, Dropout
	from keras.models import Sequential
	# from keras.optimizers import RMSprop

	model = Sequential()

	model.add(LSTM(512, input_shape=(1,3), return_sequences=True))
	model.add(Dropout(0.5))

	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.5))

	model.add(LSTM(512))
	model.add(Dropout(0.5))	

	model.add(Dense(datas_Y.shape[1], activation='softmax'))

	opt = keras.optimizers.Adam(lr=0.001)

	model.compile(
			loss='categorical_crossentropy',
			optimizer= opt,
			metrics= ['accuracy']
		)

	model.fit(X_data,datas_Y,epochs=6000)
	model.save("Example_code_weight_2.h5")
