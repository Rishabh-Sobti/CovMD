import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import cv2


def _load_label_names():
    """
    Load the label names from file
    """
    return ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening', 'Cardiomegaly','Emphysema','Edema','Fibrosis','Pneumonia','Hernia']

def load_batch(dataset_folder_path,batch_no): #loading batches from the files
    x=[]
    y=[]
    i=-1
    labels=_load_label_names()
    with open('sample/sample_labels.csv') as cf:
        reader=csv.reader(cf)
        for row in reader:
            i=i+1
            if(i==0):
                continue
            if(i<(batch_no-1)*450):
                continue
            elif(i>=batch_no*450):
                break
            img=row[0]
            y_this=[]
            corr_labels=row[1].split('|')
            for cl in corr_labels:
                y_this.append(labels.index(cl))
            y.append(y_this)
            full_size_image = cv2.imread('sample/images/'+img)
            x.append(cv2.resize(full_size_image, (256,256), interpolation=cv2.INTER_CUBIC))
    return x,y

def load_labels(batch_no, one_hot_encode): #loading only the labels from files
    y=[]
    i=-1
    labels=_load_label_names()
    with open('sample/sample_labels.csv') as cf:
        reader=csv.reader(cf)
        for row in reader:
            i=i+1
            if(i==0):
                continue
            if(i<(batch_no-1)*450):
                continue
            elif(i>=batch_no*450):
                break
            y_this=[]
            corr_labels=row[1].split('|')
            for cl in corr_labels:
                y_this.append(labels.index(cl))
            y.append(y_this)
    return one_hot_encode(y)

def display_stats(dataset_folder_path, sample_no):
    """
    Display Stats of the the dataset
    """
    info=None
    if not (0 < sample_no <= 5606):
        print('{} sample is out of range.'.format(sample_no))
        return None
    with open('sample/sample_labels.csv') as file:
        reader=csv.reader(file)
        for idx,row in enumerate(reader):
            if idx==sample_no:
                info=row
                break
    print('\nExample of Image {}:'.format(sample_no))
    print('Id of image: {}'.format(info[0]))
    print('Finding Labels: {}'.format(info[1]))
    print('Age: {}      \nSex: {}'.format(info[4],info[5]))
    img=mpimg.imread('sample/images/'+info[0])
    plt.imshow(img,cmap='gray')
    plt.show()
    #plt.imshow(sample_image)


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename): 
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 13
    valid_features = []
    valid_labels = []
    test_features=[]
    test_labels=[]

    for batch_i in range(1, n_batches + 1):
        features, labels = load_batch(dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)
        test_count = int(len(features)*0.2)
        tot_count = validation_count + test_count

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            np.array(features[:-tot_count]),
            np.array(labels[:-tot_count]),
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        test_features.extend(features[-tot_count:-validation_count])
        test_labels.extend(labels[-tot_count:-validation_count])
        
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')
    
    
    # Preprocess and Save all training data in 2 files
    for batch in range(1,3):
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            np.array(test_features[int((len(test_features)*(batch-1))/2):int((len(test_features)*batch)/2)]),
            np.array(test_labels[int((len(test_labels)*(batch-1))/2):int((len(test_labels)*batch)/2)]),
            'preprocess_training_'+str(batch)+'.p')


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)
	
def load_valid_batch(batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_validation.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


