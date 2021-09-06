'''             Knowledge Distillation Through Supervised Contrastive Feature/Representation Approximation          '''

#Importing Libraries

import os, time, argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
import warnings

#For GPU and Random seed setting

tf.debugging.set_log_device_placement(False)
warnings.filterwarnings("ignore")
tf.random.set_seed(666)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #change it with gpu id number("-1" for CPU)

#Defining Hyperparameters

num_classes = 100
input_shape = (32, 32, 3)
batch_size =128 #Change according to available space
hidden_units = 256
projection_units = 128
num_epochs = 240
dropout_rate = 0.25
temperature = 0.05

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

#Applying Data Augmentation

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.02),
        layers.experimental.preprocessing.RandomWidth(0.2),
        layers.experimental.preprocessing.RandomHeight(0.2),
    ]
)

# Setting the state of the normalization layer.
data_augmentation.layers[0].adapt(x_train)

#Callbacks Settings (Checkpointing and Scheduler)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar100_%s_model.{epoch:03d}.h5' 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

#Learning Rate Scheduler

def lr_schedule(epoch):
  """Learning Rate Schedule

  Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
  Called automatically every epoch as part of callbacks during training.

  # Arguments
      epoch (int): The number of epochs

  # Returns
      lr (float32): learning rate
  """
  lr = 1e-3
  if epoch > 180:
      lr *= 0.5e-3
  elif epoch > 160:
      lr *= 1e-3
  elif epoch > 120:
      lr *= 1e-2
  elif epoch > 80:
      lr *= 1e-1
  print('Learning rate: ', lr)
  return lr

checkpoint = ModelCheckpoint(filepath=filepath,
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
distiller_callbacks = [lr_reducer, lr_scheduler]

#Define Teacher and Student Encoders

#Change the Models below (teacher and student) to conduct different experiments and 
#refer url: https://keras.io/api/applications/
#and the Model folder to select different Models

def teacher_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=(256,256,3), pooling="avg"
    )
    for layer in resnet.layers:
        layer.trainable = True
  
    inputs = keras.Input(shape=input_shape)
    resize = layers.UpSampling2D()(inputs)
    resize = layers.UpSampling2D()(resize)
    resize = layers.UpSampling2D()(resize)
    augmented = data_augmentation(resize)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="teacher_encoder")
    return model


teacher_encoder = teacher_encoder()
teacher_encoder.summary()

def student_encoder():
    mobilenet = keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(256,256,3), pooling="avg"
    )
    for layer in mobilenet.layers:
        layer.trainable = True
    inputs = keras.Input(shape=input_shape)
    resize = layers.UpSampling2D()(inputs)
    resize = layers.UpSampling2D()(resize)
    resize = layers.UpSampling2D()(resize)
    augmented = data_augmentation(resize)
    outputs = mobilenet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="student_encoder")
    return model


student_encoder = student_encoder()
student_encoder.summary()

#Creating Classifier

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.BatchNormalization()(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# Defining Supervised Contrastive Loss from https://keras.io/examples/vision/supervised-contrastive-learning/

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# Defining projection Head

def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
    )
    return model

#Attaching Projection Head with Encoder

teacher_encoder_with_projection_head = add_projection_head(teacher_encoder)
student_encoder_with_projection_head = add_projection_head(student_encoder)

#Compiling the Teacher Encoder

teacher_encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
    loss=SupervisedContrastiveLoss(temperature),
)

teacher_encoder_with_projection_head.summary()

#Training the Teacher Model with SCL

history = teacher_encoder_with_projection_head.fit(
    x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=2,
     validation_data=(x_test,y_test),
     callbacks=callbacks)

#Classification with Frozen Encoder

teacher_classifier = create_classifier(teacher_encoder, trainable=False)
history = teacher_classifier.fit(x=x_train, y=y_train,validation_data=(x_test,y_test), batch_size=batch_size, 
    epochs=num_epochs,verbose=2, shuffle=True, callbacks=callbacks)

teacher_accuracy = teacher_classifier.evaluate(x_test, y_test)[1]

#Defining Our Proposed Distillation Method 

class SCL_Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(SCL_Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        student_loss_fn,
        similarity_fn,
        distillation_loss_fn,
        alpha=0.1,
        beta=0.7,
        temperature=3,
        temp=5,
    ):
        super(SCL_Distiller, self).compile(optimizer=optimizer, metrics=[])
        self.student_loss_fn = student_loss_fn
        self.similarity_fn =similarity_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.temp=temp

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            similarity_loss = self.similarity_fn(
                tf.nn.softmax(teacher_predictions / self.temp, axis=1),
                tf.nn.softmax(student_predictions / self.temp, axis=1),
            )

            #Overall Proposed Loss
            loss = self.alpha * student_loss + (1 - self.alpha-self.beta) * distillation_loss+ self.beta * similarity_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss,"similarity_loss": similarity_loss}
        )
        return results

# Initialize and Compile Distiller
SCL_Distiller = SCL_Distiller(student=student_encoder_with_projection_head, teacher=teacher_encoder_with_projection_head)
SCL_Distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
    student_loss_fn=SupervisedContrastiveLoss(temperature),
    similarity_fn=tf.keras.losses.CosineSimilarity(axis=1),
    distillation_loss_fn=keras.losses.KLDivergence(),
    #Change The following hyper-parameters for fine-tuning
    alpha=0.05,
    temperature=10,
    beta=0.9,
    temp=10,
)

# Distill teacher encoder to student encoder
SCL_Distiller.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=2,
    callbacks=distiller_callbacks,shuffle=True)

#Extracting the student encoder's weights from SCL_Distiller and loading to the student encoder model

student_weight=SCL_distiller.layers[1].get_weights()
student_encoder_with_projection_head.set_weights(student_weight)

#Attaching the classifier to the student Backbone network

student_classifier = create_classifier(student_encoder, trainable=False)

history = student_classifier.fit(x=x_train, y=y_train,validation_data=(x_test,y_test), batch_size=batch_size, 
    epochs=num_epochs, verbose=2, callbacks=callbacks, shuffle=True)

student_accuracy = student_classifier.evaluate(x_test, y_test)[1]

#Printing the final result

print(f"Test accuracy(Teacher): {round(teacher_accuracy * 100, 2)}%")
print(f"Test accuracy(Student): {round(student_accuracy * 100, 2)}%")

 '''                                                  *** END ***                                                  '''