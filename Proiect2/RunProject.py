import numpy as np

from Proiect2.FacialDetector import *
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
path_model = "../data/salveazaFisiere/best_cnn_model.pth"
if not os.path.exists(path_model):
    fete_extrase_path = "salveaza/fete"
    if os.path.exists(fete_extrase_path):
        print('Exista poze incarcate')
    else:
        os.makedirs(fete_extrase_path)
        print('Incarc poze pozitive:')
        facial_detector.get_positive_descriptors()

    negative_extrase_path = "salveaza/negative"
    if os.path.exists(negative_extrase_path):
        print('Exista negative incarcate')
    else:
        os.makedirs(negative_extrase_path)
        print('Incarc poze pozitive:')
        facial_detector.get_negative_descriptors()

    def preprocess_image(img):
        # Convert to grayscale
        img = img / 255.0
        # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Shape becomes (1, 36, 36)
        return img

    def load_images_from_folder():
        pozitive = []
        negative = []
        folder = "salveaza/fete"
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = cv.imread(img_path)
                if img is not None:
                    pozitive.append(img)
        folder = "salveaza/negative"
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = cv.imread(img_path)
                if img is not None:
                    negative.append(img)
        return pozitive, negative


    pozitive, negative = load_images_from_folder()
    pozitive2, negative2 = [], []
    for pozitiv in pozitive:
        pozitiv = preprocess_image(pozitiv)
        pozitive2.append(pozitiv)
    for negativ in negative:
        negativ = preprocess_image(negativ)
        negative2.append(negativ)



    # # Pasul 4. Invatam clasificatorul liniar
    training_examples = np.concatenate((np.squeeze(np.squeeze(pozitive2)), np.squeeze(np.squeeze(negative2))), axis=0)
    inputs = torch.tensor(training_examples, dtype=torch.float32)
    # Ensure input tensor shape is [batch_size, channels, height, width]
    inputs = inputs.permute(0, 3, 1, 2)  # Move the channels to the second dimension
    #print(inputs.shape)
    train_labels = np.concatenate((np.ones(len(pozitive2)), np.zeros(len(negative2))))
    facial_detector.train_classifier(inputs, train_labels)

detections, scores, predicted_classes, confidences, file_names = facial_detector.run()
#print(scores)

#print(detections)
detections_dad =np.empty((0, 4), dtype=float)
detections_deedee =np.empty((0, 4), dtype=float)
detections_dexter = np.empty((0, 4), dtype=float)
detections_mom =np.empty((0, 4), dtype=float)
file_names_dad = np.array([])
file_names_deedee = np.array([])
file_names_dexter = np.array([])
file_names_mom = np.array([])
scores_dad = np.array([])
scores_deedee = np.array([])
scores_dexter = np.array([])
scores_mom = np.array([])
np_detection = np.empty((0, 4), dtype=float)
np_scores = np.array([])
np_file_names = np.array([])
#(detections)


for i in range(len(detections)):
    detection = detections[i]
    #print(detection)
    score = scores[i]
    predicted_class = predicted_classes[i]
    confidence = confidences[i]
    file_name = file_names[i]

    np_detection = np.concatenate((np_detection, np.array([detection])), axis=0)
    np_scores = np.append(np_scores, score)
    np_file_names = np.append(np_file_names, file_name)

    if predicted_class == 0:
        character = "dad"

        detections_dad = np.concatenate((detections_dad, np.array([detection])), axis=0)
        file_names_dad = np.append(file_names_dad, file_name)
        scores_dad = np.append(scores_dad, confidence)
    elif predicted_class == 1:
        character = "deedee"

        detections_deedee = np.concatenate((detections_deedee, np.array([detection])), axis=0)
        file_names_deedee = np.append(file_names_deedee, file_name)
        scores_deedee = np.append(scores_deedee, confidence)

    elif predicted_class == 2:
        character = "dexter"

        detections_dexter = np.concatenate((detections_dexter, np.array([detection])), axis=0)
        file_names_dexter = np.append(file_names_dexter, file_name)
        scores_dexter = np.append(scores_dexter, confidence)

    elif predicted_class == 3:
        character = "mom"

        detections_mom = np.concatenate((detections_mom, np.array([detection])), axis=0)
        file_names_mom = np.append(file_names_mom, file_name)
        scores_mom = np.append(scores_mom, confidence)

    else:
        continue

        # np_predicted_classes = np.concatenate((np_predicted_classes, predicted_class), axis=0)
        # np_confidence = np.concatenate((np_confidence, confidence), axis=0)


# print(f"final {np_detection}")

file_path = "Solutie/331_Chitu_Tudor/Task1/detections_all_faces.npy"  # Change this to your desired file path
np.save(file_path, np_detection)
file_path = "Solutie/331_Chitu_Tudor/Task1/file_names_all_faces.npy"  # Change this to your desired file path
np.save(file_path, np_file_names)
file_path = "Solutie/331_Chitu_Tudor/Task1/scores_all_faces.npy"  # Change this to your desired file path
np.save(file_path, np_scores)

file_path = "Solutie/331_Chitu_Tudor/Task2/detections_dad.npy"  # Change this to your desired file path
np.save(file_path, detections_dad)
file_path = "Solutie/331_Chitu_Tudor/Task2/file_names_dad.npy"  # Change this to your desired file path
np.save(file_path, file_names_dad)
file_path = "Solutie/331_Chitu_Tudor/Task2/scores_dad.npy"  # Change this to your desired file path
np.save(file_path, scores_dad)

file_path = "Solutie/331_Chitu_Tudor/Task2/detections_deedee.npy"  # Change this to your desired file path
np.save(file_path, detections_deedee)
file_path = "Solutie/331_Chitu_Tudor/Task2/file_names_deedee.npy"  # Change this to your desired file path
np.save(file_path, file_names_deedee)
file_path = "Solutie/331_Chitu_Tudor/Task2/scores_deedee.npy"  # Change this to your desired file path
np.save(file_path, scores_deedee)

file_path = "Solutie/331_Chitu_Tudor/Task2/detections_dexter.npy"  # Change this to your desired file path
np.save(file_path, detections_dexter)
file_path = "Solutie/331_Chitu_Tudor/Task2/file_names_dexter.npy"  # Change this to your desired file path
np.save(file_path, file_names_dexter)
file_path = "Solutie/331_Chitu_Tudor/Task2/scores_dexter.npy"  # Change this to your desired file path
np.save(file_path, scores_dexter)

file_path = "Solutie/331_Chitu_Tudor/Task2/detections_mom.npy"  # Change this to your desired file path
np.save(file_path, detections_mom)
file_path = "Solutie/331_Chitu_Tudor/Task2/file_names_mom.npy"  # Change this to your desired file path
np.save(file_path, file_names_mom)
file_path = "Solutie/331_Chitu_Tudor/Task2/scores_mom.npy"  # Change this to your desired file path
np.save(file_path, scores_mom)

#
# if params.has_annotations:
#     facial_detector.eval_detections(detections, scores, file_names)
#     show_detections_with_ground_truth(detections, scores, predicted_classes, confidences, file_names, params)
# else:
#     show_detections_without_ground_truth(detections, scores, predicted_classes, confidences, file_names, params)
