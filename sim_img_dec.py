import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

def extract_features(model, img_data):
    features = model.predict(img_data)
    return features.flatten()

base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
image_directory = 'path_to_car_image_database'
image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]

features_list = []
for img_path in image_paths:
    img_data = load_and_preprocess_image(img_path)
    features = extract_features(base_model, img_data)
    features_list.append(features)

feature_matrix = np.array(features_list)
np.save('car_feature_matrix.npy', feature_matrix)
np.save('car_image_paths.npy', image_paths)

def find_similar_images(query_img_path, feature_matrix, image_paths, top_n=1, threshold=90):
    query_img_data = load_and_preprocess_image(query_img_path)
    query_features = extract_features(base_model, query_img_data)
    similarities = cosine_similarity([query_features], feature_matrix)
    sorted_indices = np.argsort(similarities[0])[::-1][:top_n]
    similarity_percentages = similarities[0][sorted_indices] * 100
    is_from_database = similarity_percentages[0] >= threshold
    return sorted_indices, similarity_percentages, is_from_database

def display_query_and_similar_images(query_img_path, similar_indices, image_paths, similarities, is_from_database):
    plt.figure(figsize=(10, 5))
    query_img = image.load_img(query_img_path, target_size=(224, 224))
    plt.subplot(1, 2, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    if is_from_database:
        similar_img = image.load_img(image_paths[similar_indices[0]], target_size=(224, 224))
        similarity_percentage = similarities[0]
        plt.subplot(1, 2, 2)
        plt.imshow(similar_img)
        plt.title(f"Match\nSimilarity: {similarity_percentage:.2f}%")
        plt.axis('off')
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, "No similar image found in the database",
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.axis('off')
    plt.show()

query_image_path = 'path_to_new_car_image.jpg'
top_n = 1
threshold = 90
similar_indices, similarities, is_from_database = find_similar_images(query_image_path, feature_matrix, image_paths, top_n, threshold)
display_query_and_similar_images(query_image_path, similar_indices, image_paths, similarities, is_from_database)