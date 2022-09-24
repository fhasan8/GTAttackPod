import sys
sys.path.append(".")
from attacks import *
from datasets import *
from models import *
import time
from keras.preprocessing.image import save_img
import os

if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    model = CIFAR10_densenet40(rel_path='./')
    print(len(model.layers))
    X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
                                                                                                       model=model,
                                                                                                       balanced=True,
                                                                                                       num_examples=100)

    l = get_first_n_examples_id_each_class(Y_test, 2)
    fgsm = Attack_FastGradientMethod(eps=0.0156)
    time_start = time.time()
    X_test_adv = fgsm.attack(model, X_test, Y_test)
    dur_per_sample = (time.time() - time_start) / len(X_test_adv)

    for i in l:
        save_img(os.path.join("./pic/", str(i)+"1.jpeg"), X_test[i])
        save_img(os.path.join("./pic/",str(i)+"2.jpeg"), X_test_adv[i])
        print(Y_test[i])
        print("----")
    print(X_test.shape)
    print(Y_test.shape)
    print(X_test_adv.shape)
    # Evaluate the adversarial examples.
    print("\n---Statistics of FGSM Attack (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=True)
