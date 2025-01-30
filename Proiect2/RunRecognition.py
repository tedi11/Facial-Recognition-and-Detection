from Proiect2.FacialDetector import *
#from Proiect2.FacialRecognition import *
from Visualize import *


params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 6713  # numarul exemplelor pozitive
params.number_negative_examples = 10000  # numarul exemplelor negative

params.threshold = 2.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

def preprocess_image(img):
    # Convert to grayscale
    img = img / 255.0
    # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Shape becomes (1, 36, 36)
    return img

def load_images_from_folder():
    dad = []
    deedee = []
    dexter = []
    mom = []
    negative = []
    characters = ["dad", "deedee", "dexter", "mom"]
    for character in characters:
        folder = f"salveaza/characters/{character}/"
        for filename in os.listdir(f"{folder}"):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = cv.imread(img_path)
                if img is not None:
                    if character == "dad":
                        dad.append(img)
                    elif character == "deedee":
                        deedee.append(img)
                    elif character == "dexter":
                        dexter.append(img)
                    elif character == "mom":
                        mom.append(img)
    folder = "salveaza/negative"
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = cv.imread(img_path)
            if img is not None:
                negative.append(img)
    return dad, deedee, dexter, mom, negative


dad, deedee, dexter, mom, negative = load_images_from_folder()
dad2, deedee2, dexter2, mom2, negative2 = [], [], [], [], []
for deedeex in deedee:
    deedeex = preprocess_image(deedeex)
    deedee2.append(deedeex)
for dadx in dad:
    dadx = preprocess_image(dadx)
    dad2.append(dadx)
for dexterx in dexter:
    dexterx = preprocess_image(dexterx)
    dexter2.append(dexterx)
for momx in mom:
    momx = preprocess_image(momx)
    mom2.append(momx)
for negativ in negative:
    negativ = preprocess_image(negativ)
    negative2.append(negativ)
print(len(dad2))



# # Pasul 4. Invatam clasificatorul liniar
training_examples = np.concatenate((np.squeeze(np.squeeze(dad2)), np.squeeze(np.squeeze(deedee2)), np.squeeze(np.squeeze(dexter2)), np.squeeze(np.squeeze(mom2)), np.squeeze(np.squeeze(negative2))), axis=0)
inputs = torch.tensor(training_examples, dtype=torch.float32)
# Ensure input tensor shape is [batch_size, channels, height, width]
inputs = inputs.permute(0, 3, 1, 2)  # Move the channels to the second dimension
print(inputs.shape)
train_labels = np.concatenate((np.zeros(len(dad2)), np.ones(len(deedee2)), np.ones(len(dexter2)) * 2, np.ones(len(mom2)) * 3, np.ones(len(negative2)) * 4))
#print(train_labels[3000])
facial_detector.train_classifier(inputs, train_labels)

#
detections, scores, recognitions, file_names = facial_detector.run()

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, recognitions, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)