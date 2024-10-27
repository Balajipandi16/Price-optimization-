### Project Title: **Demand and Pricing Optimization for E-commerce Products**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Demand Forecasting](#demand-forecasting)
   - [Pricing Optimization](#pricing-optimization)
   - [Revenue Estimation](#revenue-estimation)
4. [Model Evaluation](#model-evaluation)
5. [Technologies Used](#technologies-used)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Overview
This project aims to optimize demand forecasting, pricing strategies, and revenue estimation for various product categories in an e-commerce setting. By leveraging machine learning techniques, including ARIMA and neural networks, the project seeks to enhance pricing decisions based on elasticity, competitive pricing, and customer demand.

## Installation
To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Balajipandi16/Price-optimization-
   cd Price-optimization-
   ```

2. **Install required packages**:
   Ensure you have Python installed (preferably Python 3.7+). Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Install additional libraries**:
   If necessary, install additional libraries:
   ```bash
   pip install pandas numpy scikit-learn tensorflow statsmodels shap
   ```

## Usage
### Data Preparation
Prepare your dataset by loading it into a pandas DataFrame. Ensure that your dataset contains the following columns:
- `product_id`
-`product_category_name`
- `month_year`
- `qty`
- `total_price`
- `freight_price`
- `unit_price`
- `product_name_lenght`
- `product_description_lenght`
- `product_photos_qty`
- `product_weight_g`
- `product_score`
- `customers`
- `weekday`
- `weekend`
- `holiday`
- `month`
- `year`
- `s`
- `volume`
- `comp_1`
- `ps1`
- `fp1`
- `comp_2`
- `ps2`
- `fp2`
- `comp_3`
- `ps3`
- `fp3`
- `lag_price`
- `demand`
- `revenue_comp1`
- `profit_comp1`
- `margin_comp1`
- `revenue_comp2`
- `profit_comp2`
- `margin_comp2`
- `revenue_comp3`
- `profit_comp3`
- `margin_comp3`
- `revenue_our_product`
- `profit_our_product`
- `margin_our_product`
- `fp1_diff`
- `fp2_diff`
- `fp3_diff`
- `price_ratio_1`
- `price_ratio_2`
- `price_ratio_3`
- `price_diff_1`
- `price_diff_2`
- `price_diff_3`
- `price_change`


### Demand Forecasting
Use ARIMA and XGBoost models to forecast demand for each product category. The following steps summarize the process:
1. Check for stationarity of the demand time series.
2. Fit ARIMA models for each category.
3. Evaluate model performance using metrics like RMSE and MAE.

### Pricing Optimization
Implement a neural network model to optimize:
- `freight_price`
- `total_price`
- `unit_price`
- `price elasticity`

### Key Components Of Neural Network 
 We are using a Feedforward Neural Network (FNN), specifically a Multi-Layer Perceptron (MLP), to optimize both total_price and unit_price. Here's a breakdown of the neural network architecture and components used in the code:

Key Components of the Neural Network
Sequential Model:

The neural network is built using the Sequential model from Keras, which allows you to stack layers linearly.
Dense Layers:

Input Layer: The first Dense layer specifies the number of input features (the shape is determined by X_train_total.shape[1]).
Hidden Layers:
- The network includes multiple hidden layers with the following configurations:
- The first hidden layer has 64 neurons and uses the ReLU (Rectified Linear Unit) activation function.
- The second hidden layer has 32 neurons with ReLU activation.
- The third hidden layer has 16 neurons with ReLU activation.
- Output Layer: A single neuron output layer without an activation function, suitable for regression tasks, as it predicts continuous values (total price or unit price).
Dropout Layers:

Dropout is used after each of the first two hidden layers, which helps prevent overfitting by randomly setting a fraction of the input units to zero during training. In your code, it is set to 20% (Dropout(0.2)).
Loss Function and Optimizer:

The model is compiled with the mean squared error (MSE) loss function, which is commonly used for regression tasks.
The Adam optimizer is used, which is effective for training neural networks and adjusts the learning rate during training.
Summary of the Neural Network Architecture
The architecture can be summarized as follows:

Input Layer: Number of features as inputs (varies based on the selected features).
- Hidden Layer 1: 64 neurons, ReLU activation
- Dropout Layer 1: 20% dropout
- Hidden Layer 2: 32 neurons, ReLU activation
- Dropout Layer 2: 20% dropout
- Hidden Layer 3: 16 neurons, ReLU activation
Output Layer: 1 neuron (for predicting total or unit price)
Conclusion
The neural network used in your code is a Multi-Layer Perceptron (MLP) that is well-suited for the regression tasks of optimizing total price and unit price based on various input features.

The model will consider various features, including:
- Competitive prices
- Historical demand
- Profit margins

### Revenue Estimation
After optimizing prices, users can enter expected quantities for each product category to estimate potential revenue.

```python
example
Quantity for perfumery (Optimized Total Price: 412.29): 159
Quantity for watches_gifts (Optimized Total Price: 1689.66): 753
Category: perfumery, Expected Revenue: 65553.61
Category: watches_gifts, Expected Revenue: 1272317.31
```

## Model Evaluation
Evaluate the performance of your models using metrics such as:
- RMSE
- R² Score
- MAE

The evaluation results for each category are printed out for analysis.

## Technologies Used
- Python 3.x
- Libraries:
  - Pandas
  - NumPy
  - Scikit-learn
  - TensorFlow/Keras
  - Statsmodels
  - SHAP for interpretability



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to update the links, sections, and content as needed for your project. Good documentation is crucial for making your project accessible and understandable to other developers. Let me know if you need further assistance!
