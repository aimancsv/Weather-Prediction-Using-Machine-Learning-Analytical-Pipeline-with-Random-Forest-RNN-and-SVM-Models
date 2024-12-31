# Weather-Prediction-Using-Machine-Learning-Analytical-Pipeline-with-Random-Forest-RNN-and-SVM-Models
This repo uses machine learning to enhance weather forecasting accuracy with the "Weather in Szeged from 2006–2016" dataset from Kaggle. Algorithms like Random Forest, RNN, and SVM are implemented, combined with advanced data preprocessing, visualization, and evaluation, addressing traditional forecasting limitations for actionable insights.


---

#### Key Features
1. **Data Preprocessing**:
   - Data cleaning, null value handling, and transformation of categorical data into numerical formats for model compatibility.
   - Use of Python libraries such as pandas, NumPy, and Scikit-learn for efficient data manipulation.
  
     
     <img width="400" alt="Screenshot 2024-12-31 at 10 20 05 AM" src="https://github.com/user-attachments/assets/890b7c4b-f61a-4acf-8536-e60edb8b4de2" />


2. **Machine Learning Models**:
   - **Random Forest**: Demonstrated the best performance with high accuracy and interpretability.

    <img width="500" alt="Screenshot 2024-12-31 at 10 26 23 AM" src="https://github.com/user-attachments/assets/a580c9e1-1f9e-4dc6-b18f-5ede4ef94ae9" />
    <img width="500" alt="Screenshot 2024-12-31 at 10 27 24 AM" src="https://github.com/user-attachments/assets/a1577e0e-8716-4f86-8d7e-3b62bca59891" />


   - **RNN**: Utilized TensorFlow for capturing temporal patterns in weather data.

     <img width="500" alt="Screenshot 2024-12-31 at 10 28 02 AM" src="https://github.com/user-attachments/assets/b534a8c7-5514-4305-aa10-e831d392d3c7" />
     <img width="500" alt="Screenshot 2024-12-31 at 10 28 56 AM" src="https://github.com/user-attachments/assets/d26c55ee-0042-4919-9040-6b08f37693db" />


   - **SVM**: Explored for regression tasks but showed limited suitability for this dataset.
     <img width="500" alt="Screenshot 2024-12-31 at 10 29 22 AM" src="https://github.com/user-attachments/assets/cf03e4d0-9559-4817-a6d5-8f0285277dff" />
     <img width="500" alt="Screenshot 2024-12-31 at 10 29 36 AM" src="https://github.com/user-attachments/assets/cc549860-00f3-4c58-bb0d-953caf1f7535" />



3. **Evaluation Metrics**:
   - Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score for assessing model performance.
   - **Random Forest:**
     
     <img width="500" alt="Screenshot 2024-12-31 at 10 32 10 AM" src="https://github.com/user-attachments/assets/7affb4ed-c875-4013-8e1c-6bbbadfcefd0" />
     
   - **Recurrent Neural Networks (RNN):**
     
     <img width="500" alt="Screenshot 2024-12-31 at 10 32 32 AM" src="https://github.com/user-attachments/assets/dfdba764-8bbd-41cd-b261-26947e6512e5" />
     
   - **Support Vector Machines (SVM):**
     
     <img width="500" alt="Screenshot 2024-12-31 at 10 34 33 AM" src="https://github.com/user-attachments/assets/fdb54b0b-a7ff-4fa0-9f37-ae862430d870" />
     

4. **Insights and Analysis**:
   - Explored relationships between weather variables like temperature, humidity, and wind speed.
   - Identified trends in monthly temperature averages and correlations between weather conditions and temperature.
   - **Temperature:**
     <img width="500" alt="Screenshot 2024-12-31 at 10 40 04 AM" src="https://github.com/user-attachments/assets/b548e641-a9c4-4d5b-88a9-712a58043c70" />

   - **Humidity:**
     <img width="500" alt="image" src="https://github.com/user-attachments/assets/65e6cf6e-c899-4c05-9c0b-bb40f455d702" />
     
   - **Wind Speed::**
     <img width="500" alt="Screenshot 2024-12-31 at 10 41 37 AM" src="https://github.com/user-attachments/assets/0a726479-92f6-4ed3-b0af-08737bfa2f74" />
     
   - **Top 10 Weather:**
     <img width="500" alt="Screenshot 2024-12-31 at 10 42 04 AM" src="https://github.com/user-attachments/assets/360450f0-4155-479f-b272-bed6aacf9925" />

   - **Relations of Weather Summary and Temperature:**
     
     - <img width="700" alt="Screenshot 2024-12-31 at 10 42 32 AM" src="https://github.com/user-attachments/assets/2db4202a-7539-4a54-b694-3a7d154f9404" />
     
     - <img width="350" alt="Screenshot 2024-12-31 at 10 42 51 AM" src="https://github.com/user-attachments/assets/68ee6e14-8691-453a-8839-5e0afa15c16e" /> <img width="350" alt="Screenshot 2024-12-31 at 10 43 06 AM" src="https://github.com/user-attachments/assets/1a039147-fd5b-4cb1-be64-c82f4b5bcf05" /> <img width="350" alt="Screenshot 2024-12-31 at 10 43 18 AM" src="https://github.com/user-attachments/assets/01f1dedb-917d-4db4-a47d-2b4d429e6fec" /> <img width="350" alt="Screenshot 2024-12-31 at 10 43 37 AM" src="https://github.com/user-attachments/assets/1282bbaa-6c41-4a26-8bf2-f8f189bf5f5b" /> <img width="350" alt="Screenshot 2024-12-31 at 10 43 52 AM" src="https://github.com/user-attachments/assets/acc9b2c0-d43b-456b-a4ee-2d8275ab5998" />






   - **Mean Temperature by Month:**
     
     <img width="500" alt="Screenshot 2024-12-31 at 10 44 13 AM" src="https://github.com/user-attachments/assets/475c1b56-8905-4351-ab92-fea6a0c24545" />

   - **Monthly Statistic of Temperature (2006-2009):**
     
     <img width="500" alt="Screenshot 2024-12-31 at 10 44 34 AM" src="https://github.com/user-attachments/assets/dca58543-c64c-47ef-9791-e88fef70e621" />
     <img width="500" alt="Screenshot 2024-12-31 at 10 44 56 AM" src="https://github.com/user-attachments/assets/cf5315e5-48c4-4055-b609-ca94b3f30a43" />
     <img width="500" alt="Screenshot 2024-12-31 at 11 04 44 AM" src="https://github.com/user-attachments/assets/1ad06d5c-0ba7-4f13-93bd-c53edab01d08" />
     <img width="500" alt="Screenshot 2024-12-31 at 10 45 27 AM" src="https://github.com/user-attachments/assets/d7e83f48-029e-49fd-99cf-bab88ec3088a" />

---


#### Results
- **Best Model**: Random Forest, selected for its balance of accuracy, speed, and ease of interpretation.
- **Performance**: Achieved MAE and RMSE scores indicating strong prediction accuracy, though overfitting was noted in some cases.
- **Visualization**: Provided clear insights through temperature distribution plots, scatter diagrams, and learning curves.

---

#### Future Work
1. Explore advanced neural networks like LSTMs or CNNs for capturing temporal and spatial dependencies.
2. Integrate ensemble techniques (e.g., boosting or stacking) for improved robustness.
3. Conduct cross-validation on diverse datasets to ensure generalizability.
4. Implement domain-specific evaluation metrics tailored to weather forecasting.

