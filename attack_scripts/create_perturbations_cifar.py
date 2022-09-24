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
    X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
                                                                                                       model=model,
                                                                                                       balanced=True,
                                                                                                       num_examples=10)

    bim = Attack_BasicIterativeMethod(eps=0.008, eps_iter=0.0012, nb_iter=2)
    X_test_adv = bim.attack(model, X_test, Y_test)
    bim2 = Attack_BasicIterativeMethod(eps=0.008, eps_iter=0.0012, nb_iter=5)
    X_test_adv2 = bim2.attack(model, X_test, Y_test)
    bim3 = Attack_BasicIterativeMethod(eps=0.008, eps_iter=0.0012, nb_iter=10)
    time_start = time.time()
    
    
    X_test_adv3 = bim3.attack(model, X_test, Y_test)
    dur_per_sample = (time.time() - time_start) / len(X_test_adv)
    for i in range(10):
        save_img(os.path.join("./pic1/", str(i)+"1.jpeg"), X_test[i])
        save_img(os.path.join("./pic1/",str(i)+"2.jpeg"), X_test_adv[i])
        save_img(os.path.join("./pic1/",str(i)+"3.jpeg"), X_test_adv2[i])
        save_img(os.path.join("./pic1/",str(i)+"4.jpeg"), X_test_adv3[i])
        print(Y_test[i])
        print("----")
    # Evaluate the adversarial examples.
    print("\n---Statistics of BIM Attack (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=False)
