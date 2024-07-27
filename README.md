# Animal Classifier Project

This project utilizes TensorFlow and EfficientNetB7 to classify animal images from the "Animals10" dataset. The project is structured into three main scripts that prepare data, train the model, and evaluate its performance.

## Project Structure

- `scripts/`
  - `data_preparation.py`: Prepares the image data for training by organizing it into a suitable format and applying initial preprocessing.
  - `model_training.py`: Contains the model setup, training process, and saves the trained model.
  - `evaluation.py`: Evaluates the trained model on a test set and generates performance metrics.

- `datasets/`
  - `animals10/`
    - `raw-img/`: Contains subdirectories for each class of animal images.

- `models/`
  - `checkpoints/`: Stores the checkpoints during model training.

## Requirements

This project requires Python 3.8 or later and the following Python libraries:
- TensorFlow 2.x
- NumPy
- Matplotlib
- pandas

To install the required libraries, run:
```bash
pip install -r requirements.txt
```

## Running the Scripts

### Data Preparation

To prepare the data, navigate to the project root directory and run:
```bash
python -m scripts.data_preparation
```
This script organizes the data into a training/validation split and applies initial preprocessing such as resizing and normalization.

### Model Training

To train the model, run:
```bash
python -m scripts.model_training
```
This script will load the prepared data, define and compile the model, and train it while saving checkpoints.

### Model Evaluation

After training, evaluate the model's performance by running:
```bash
python -m scripts.evaluation
```
This script loads the best model checkpoint and evaluates its performance on the test data set. It also plots training and validation accuracy and loss graphs.

## Notes

Ensure that the dataset is correctly placed in the `datasets/animals10/raw-img/` directory and that you have updated the paths in the scripts if your directory structure is different.

For any issues or further instructions, please refer to the official TensorFlow documentation or raise an issue in this project's repository.
```

### Additional Tips

- **Version Control**: If you are using a version control system like Git, you might want to add `.gitignore` files to avoid uploading large data files or model binaries.
- **Licensing and Credits**: Include a section for licensing if your project or its dependencies require it, and give credits to data or code sources if necessary.
- **Further Documentation**: You might want to expand the `README.md` with sections on how to contribute to the project, detailed descriptions of each script, and more detailed setup instructions if your environment setup is complex.