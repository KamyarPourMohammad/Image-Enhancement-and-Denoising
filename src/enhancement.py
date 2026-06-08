import numpy as np
import cv2 as cv
import os


def log_transform(image):
    img_float = image.astype(np.float32)
    img_normalized = img_float / 255.0
    img_log = np.log1p(img_normalized)
    img_scaled = (img_log / np.max(img_log)) * 255.0
    img_output = np.clip(img_scaled, 0, 255).astype(np.uint8)
    return img_output


def exp_transform(image, beta):
    img_float = image.astype(np.float32)
    img_normalized = img_float / 255.0
    exp_term = np.exp(beta * img_normalized) - 1
    denominator = np.exp(beta) - 1
    img_exp = exp_term / denominator
    img_scaled = img_exp * 255.0
    img_output = np.clip(img_scaled, 0, 255).astype(np.uint8)
    return img_output


def gamma_correction(image, gamma):
    img_float = image.astype(np.float32)
    img_normalized = img_float / 255.0
    img_gamma = np.power(img_normalized, gamma)
    img_scaled = img_gamma * 255.0
    img_output = np.clip(img_scaled, 0, 255).astype(np.uint8)
    return img_output


def test_enhancement_functions(image_path, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found!")
        return None

    img_log = log_transform(img)
    img_exp_beta01 = exp_transform(img, beta=0.1)
    img_exp_beta03 = exp_transform(img, beta=0.3)
    img_exp_beta05 = exp_transform(img, beta=0.5)
    img_exp_beta001 = exp_transform(img, beta=0.01)
    img_gamma_05 = gamma_correction(img, gamma=0.5)
    img_gamma_15 = gamma_correction(img, gamma=1.5)
    img_gamma_20 = gamma_correction(img, gamma=2.0)
    img_gamma_07 = gamma_correction(img, gamma=0.7)
    img_gamma_12 = gamma_correction(img, gamma=1.2)

    cv.imwrite(os.path.join(output_dir, '01_original.jpg'), img)
    cv.imwrite(os.path.join(output_dir, '02_log_transform.jpg'), img_log)
    cv.imwrite(os.path.join(output_dir, '03_exp_beta_0.01.jpg'), img_exp_beta001)
    cv.imwrite(os.path.join(output_dir, '03_exp_beta_0.1.jpg'), img_exp_beta01)
    cv.imwrite(os.path.join(output_dir, '03_exp_beta_0.3.jpg'), img_exp_beta03)
    cv.imwrite(os.path.join(output_dir, '03_exp_beta_0.5.jpg'), img_exp_beta05)
    cv.imwrite(os.path.join(output_dir, '04_gamma_0.5_brighten.jpg'), img_gamma_05)
    cv.imwrite(os.path.join(output_dir, '04_gamma_0.7_brighten.jpg'), img_gamma_07)
    cv.imwrite(os.path.join(output_dir, '04_gamma_1.2_darken.jpg'), img_gamma_12)
    cv.imwrite(os.path.join(output_dir, '04_gamma_1.5_darken.jpg'), img_gamma_15)
    cv.imwrite(os.path.join(output_dir, '04_gamma_2.0_darken.jpg'), img_gamma_20)

    print(f"\nAll images saved successfully in '{output_dir}' folder!")
    print("\nSaved files:")
    print("  - 01_original.jpg")
    print("  - 02_log_transform.jpg")
    print("  - 03_exp_beta_0.01.jpg")
    print("  - 03_exp_beta_0.1.jpg")
    print("  - 03_exp_beta_0.3.jpg")
    print("  - 03_exp_beta_0.5.jpg")
    print("  - 04_gamma_0.5_brighten.jpg")
    print("  - 04_gamma_0.7_brighten.jpg")
    print("  - 04_gamma_1.2_darken.jpg")
    print("  - 04_gamma_1.5_darken.jpg")
    print("  - 04_gamma_2.0_darken.jpg")

    cv.imshow('Original Image', img)
    cv.imshow('Log Transform', img_log)
    cv.imshow('Exp Transform (beta=0.1)', img_exp_beta01)
    cv.imshow('Exp Transform (beta=0.3)', img_exp_beta03)
    cv.imshow('Gamma (gamma=0.5) - Brighten', img_gamma_05)
    cv.imshow('Gamma (gamma=1.5) - Darken', img_gamma_15)
    cv.imshow('Gamma (gamma=2.0) - Darken', img_gamma_20)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return {
        'original': img,
        'log': img_log,
        'exp_beta01': img_exp_beta01,
        'exp_beta03': img_exp_beta03,
        'gamma_05': img_gamma_05,
        'gamma_15': img_gamma_15,
        'gamma_20': img_gamma_20
    }


if __name__ == "__main__":
    image_path = "m35597_einstein.jpg"
    results = test_enhancement_functions(image_path, output_dir="my_results")