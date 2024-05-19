# UNQ_C2

tf.random.set_seed(1234)  # for consistent results
model = Sequential([
    tf.keras.Input(shape=(400,)),     # @REPLACE
    Dense(25, activation='relu', name = "L1"),  # @REPLACE
    Dense(15, activation='relu',  name = "L2"),  # @REPLACE
    Dense(10, activation='linear', name = "L3"),  # @REPLACE
    ], name="my_model"
)
