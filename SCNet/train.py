from scnet import *
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
# GPU setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)
KTF.set_session(sess)

def load_data():
    # load_data function should be defined by your dataset
    pass

def main():
    batch_size = 128
    epochs =300
    num_classes = 2
    wordVec = 76
    num_features = 748

    h = 224
    w = 224
    c = 3

    save_model = os.getcwd() + "/model/scnet"

    x_img_train, x_txt_train, y_train, x_img_test, x_txt_test, y_test, x_img_val, x_txt_val, y_val = load_data()

    scnet = SCNet()
    scnet.fit([x_img_train, x_txt_train], [y_train],
            num_classes,
            [(h,w,c),(wordVec,)],
            batch_size,
            epochs,
            validation_data=([x_img_val, x_txt_val], [y_val]),
            num_features=num_features
            )

    scnet.save(save_model, type='json')
    scnet.evaluate([x_img_test, x_txt_test],[y_test])
    y_score=scnet.predict([x_img_test, x_txt_test])
    print("acc:",y_score)

if __name__ == '__main__':
    main()

