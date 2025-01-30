from Parameters import *
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from Proiect2.FacialRecognition import SimpleCNN_Recognition


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):  # Default to 2 classes
        super(SimpleCNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def eval_train(self, device, training_examples, train_labels):
        print("aa")
        self.eval()  # Switch to evaluation mode

        with torch.no_grad():
            inputs = torch.tensor(training_examples, dtype=torch.float32).to(device)
            labels = torch.tensor(train_labels, dtype=torch.long).to(device)

            outputs = self(inputs)  # Get raw outputs (logits)
            scores = (outputs[:, 1] - outputs[:, 0]).cpu().numpy()  # Compute the difference between output neurons

            positive_scores = scores[train_labels > 0]
            negative_scores = scores[train_labels <= 0]

            # Plot the decision scores
            plt.plot(np.sort(positive_scores), label='Scores positive examples')
            plt.plot(np.zeros(len(positive_scores)), label='0')
            plt.plot(np.sort(negative_scores), label='Scores negative examples')
            plt.xlabel('Training Example Index')
            plt.ylabel('Classifier Score (Difference of Output Neurons)')
            plt.title('Distribution of Classifier Scores on Training Examples')
            plt.legend()
            plt.show()



def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=5, fy=5)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def intersection_over_union( bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

transform = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate images by ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.ToTensor()  # Convert PIL image to tensor
])

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None



    def get_positive_descriptors(self):

        characters = ["dad", "dexter", "mom", "deedee"]
        positive_descriptors = []
        contor = 1
        for character in characters:
            images_path = os.path.join(f"antrenare/{character}", '*.jpg')
            addontation_file = open (f"antrenare/{character}_annotations.txt", "r")
            addontation_lines = addontation_file.readlines()

            files = glob.glob(images_path)
            num_images = len(files)

            print('Calculam descriptorii pt %d imagini pozitive...' % num_images)

            for i in range(num_images):

                img_no = "000" + str(i+1)
                img_no = img_no[-4:]
                #print(img_no)
                print('Procesam exemplul pozitiv numarul %d...' % i)

                for line in addontation_lines:
                    line_features = line.split()
                    if line_features[0][:4] == img_no:
                        #(line_features)
                        img = cv.imread(files[i])
                        cropped_img = img[int(line_features[2]):int(line_features[4]), int(line_features[1]):int(line_features[3])]
                        cropped_img = cv.resize(cropped_img, (64, 64))
                        flipped_img = cv.flip(cropped_img, flipCode=1)
                        #cropped_img = np.float32(cropped_img / 255.0)
                        # cropped_img = np.expand_dims(cropped_img, axis=0)
                        savefile = f"salveaza/fete/{contor}.jpg"
                        cv.imwrite(savefile, cropped_img)
                        contor = contor + 1
                        #TODO salveaza imaginea in folder

                        if self.params.use_flip_images:
                            savefile = f"salveaza/fete/{contor}.jpg"
                            #flipped_img = np.float32(flipped_img / 255.0)
                            # flipped_img = np.expand_dims(flipped_img, axis=0)
                            cv.imwrite(savefile, flipped_img)
                            contor = contor + 1
                            # TODO salveaza imaginea in folder

        #positive_descriptors = np.array(positive_descriptors)
        #print(positive_descriptors.shape)
        #return positive_descriptors

#TODO sa fac flipuite imaginile, sa verific coordonatele daca sunt date bine la poz neg

    def get_negative_descriptors(self):

        characters = ["dad", "dexter", "mom", "deedee"]
        contor = 1
        negative_descriptors = []
        for character in characters:
            images_path = os.path.join(f"antrenare/{character}", '*.jpg')
            addontation_file = open(f"antrenare/{character}_annotations.txt", "r")
            addontation_lines = addontation_file.readlines()
            files = glob.glob(images_path)
            num_images = len(files)
            num_negative_per_image = 13
            print(num_negative_per_image)

            print('Calculam descriptorii pt %d imagini negative' % num_images)
            for i in range(num_images):
                print('Procesam exemplul negativ numarul %d...' % i)
                img = cv.imread(files[i])
                # TODO: iau o marime random, verific daca se suprapune si cu cat, redimensionez
                num_rows = img.shape[0]
                num_cols = img.shape[1]
                window_dim = np.random.randint(low=50, high=200, size=num_negative_per_image)
                x = np.random.randint(low=1, high=num_cols - 201, size=num_negative_per_image)
                y = np.random.randint(low=1, high=num_rows - 201, size=num_negative_per_image)
                img_no = "000" + str(i + 1)
                img_no = img_no[-4:]

                for idx in range(len(y)):
                    ok = 1
                    for line in addontation_lines:
                        line_features = line.split()
                        if line_features[0][:4] == img_no:
                            coord_addnotation = [int(line_features[1]), int(line_features[2]), int(line_features[3]), int(line_features[4])]
                            coord_parch = [x[idx], y[idx], x[idx] + window_dim[idx], y[idx] + window_dim[idx]]
                            #print(coord_addnotation)

                            if intersection_over_union(coord_addnotation, coord_parch) > 0.05:
                                ok = 0
                    if ok == 1:
                        patch = img[y[idx]: y[idx] + window_dim[idx], x[idx]: x[idx] + window_dim[idx]]
                        patch = cv.resize(patch, (64, 64))
                        #patch = np.float32(patch / 255.0)
                        #show_image("patch", patch)
                        savefile = f"salveaza/negative/{contor}.jpg"
                        cv.imwrite(savefile, patch)
                        contor = contor + 1

        # negative_descriptors = np.array(negative_descriptors)
        # return negative_descriptors



    def train_classifier(self, training_examples, train_labels):
        cnn_file_name = os.path.join(self.params.dir_save_files, 'best_cnn_model.pth')

        # Check if a trained model exists
        if os.path.exists(cnn_file_name):
            self.best_model = torch.load(cnn_file_name, weights_only=True)
            return

        # Prepare data
        batch_size = 64
        num_classes = len(set(train_labels))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert data to PyTorch tensors and create DataLoader
        train_dataset = CustomImageDataset(torch.tensor(training_examples, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define model, loss function, and optimizer
        model = SimpleCNN(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 30
        best_accuracy = 0.0
        accuracy_list = []  # List to store accuracy per epoch

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)  # Add channel dimension if needed
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate accuracy
            accuracy = correct / total
            accuracy_list.append(accuracy)  # Append accuracy for this epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / total:.4f}, Accuracy: {accuracy:.4f}")

            # Save the best model

            torch.save(model.state_dict(), cnn_file_name)

            if (epoch+1) % 3 == 0:
                #print("aa")
                model.eval_train(device=device, training_examples=training_examples, train_labels=train_labels)

        self.best_model = model

        # Visualize training accuracy
        plt.plot(range(1, num_epochs + 1), accuracy_list, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy vs Epochs')
        plt.legend()
        plt.show()




    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]
        #print(sorted_scores)

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.1
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        """
        Modified to use SimpleCNN to calculate scores for each image region.
        """
        import os
        import timeit
        import numpy as np
        import glob
        import cv2 as cv
        from skimage.feature import hog
        import ntpath
        import torch
        import torchvision.transforms as transforms

        # Initialize the CNN model (ensure it’s loaded with pre-trained weights if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN(num_classes=2).to(device)
        state_dic = torch.load('../data/salveazaFisiere/best_cnn_model.pth')
        model.load_state_dict(state_dic)
        model.eval()  # Set the model to evaluation mode

        model_recognition = SimpleCNN_Recognition(num_classes=5).to(device)
        state_dic2 = torch.load('../data/salveazaFisiere/best_cnn_model_characters.pth')
        model_recognition.load_state_dict(state_dic2)
        model_recognition.eval()

        # Transform to preprocess the cropped regions
        preprocess = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((64, 64)),  # Resize to match CNN input size
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array with all detections
        scores = np.array([])  # array with all scores
        predicted_classes = np.array([])
        confidences = np.array([])
        file_names = np.array([])  # array with file names

        num_test_images = len(test_files)
        #num_test_images = 5

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print(f'Processing test image {i + 1}/{num_test_images}...')
            img = cv.imread(test_files[i])
            #img = np.float32(img / 255.0)
            img_size_h, img_size_w = img.shape[:2]
            image_scores = []
            image_detections = []

            # scale = 0.8
            # for j in range(1, 102):
            #     scale *= 0.98
            scale = 0.7
            for j in range(1, 48):
                scale *= 0.96
                #print(scale)


                height = int(img_size_h * scale)
                width = int(img_size_w * scale)
                img_copy = cv.resize(img, (width, height))
                #show_image("aa", img_copy)
                num_cols = img_copy.shape[1] - 1
                num_rows = img_copy.shape[0] - 1
                # num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1
                #print(j)
                for y in range(0, num_rows - 64, 5):
                    for x in range(0, num_cols - 64, 5):
                        # Crop the image region
                        x_min = x
                        y_min = y
                        x_max = int(x + 64)
                        y_max = int(y + 64)
                        crop = img_copy[y_min:y_max, x_min:x_max]
                        crop = cv.resize(crop, (64, 64))
                        if y == 0 and x == 0:
                            #show_image("test", crop)
                            pass


                        # Ensure the crop matches the input size of the CNN
                        if crop.shape[0] == 0 or crop.shape[1] == 0:
                            continue  # Skip invalid crops

                        # Preprocess the cropped image
                        crop_tensor = preprocess(crop).unsqueeze(0).to(device)


                        # Get the score from the CNN
                        with torch.no_grad():
                            cnn_output = model(crop_tensor)
                            #print(cnn_output)
                            score = (cnn_output[:, 1] - cnn_output[:, 0]).cpu().numpy()[0]
                            #print((cnn_output[:, 1] - cnn_output[:, 0]).cpu().numpy())
                            # print(score.shape)

                        if score > 5:  # You can set a threshold if needed
                            x_min = int(x_min * (1 / scale))
                            y_min = int(y_min * (1 / scale))
                            x_max = int(x_max * (1 / scale))
                            y_max = int(y_max * (1 / scale))
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)



            # Non-maximal suppression
            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(
                    np.array(image_detections), np.array(image_scores), img.shape
                )

            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))

                for detec in image_detections:

                    new_crop = img[detec[1]:detec[3], detec[0]:detec[2]]
                    new_crop = cv.resize(new_crop, (64, 64))
                    crop_tensor = preprocess(new_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        cnn_recognition_output = model_recognition(crop_tensor)
                        probabilities = torch.softmax(cnn_recognition_output, dim=1).cpu().numpy()[0]
                        face_probs = probabilities[:-1]
                        predicted_class = face_probs.argmax()
                        #print(predicted_class)
                        confidence = face_probs[predicted_class]

                        predicted_classes = np.append(predicted_classes, predicted_class)
                        confidences = np.append(confidences, confidence)
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)
            #print('6')
            end_time = timeit.default_timer()
            print(f'Time for processing test image {i + 1}/{num_test_images}: {end_time - start_time:.3f} sec.')



        return detections, scores, predicted_classes, confidences, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int64)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):

                print(gt_detections_on_image)
                overlap = intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        #plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()

