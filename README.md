# Loan Approval Prediction

A predicting model integrated into full-stack application.

## 📌 Description

A full-stack application that predicts whether a loan application will be approved based on applicant data.
The project combines **machine learning model trained in Python** with a **React frontend** and **Laravel backend**.

The Python model is integrated into the backend and frontend to provide prediction output. User can input loan application details through the React interface, which communicates with the Laravel API to retrieve predictions from the trained model.

## 🧠 Model Performance

| Model             | Accuracy | F1-Score |
|-------------------|----------|----------|
| Decision Tree     | 0.77     | 0.84     |
| Random Forest     | 0.82     | 0.89     |
| XGBoost           | 0.83     | 0.89     |


## 🚀 Features

- Instant loan approval predictions using a trained ML model

- Interactive user interface built with React.js

- Backend API through Laravel to manage inputs and serve prediction

- Integration of Python model with both frontend and backend

- Data processing and validation to ensure accurate predictions

## 🛠️ Tech stack

**Frontend:** React, Tailwind CSS, Shadcn

**Backend:** Laravel + Sanctum

**Database:** MySQL

**Machine Learning:** Python, Scikit-learn, Pandas, Numpy

**Tools:** Git, GitHub

## ⚙️ Installations & Setups

### 1️⃣ Prerequisites

Before cloning and running this project, ensure you have the following installed:

- Python 3.8 or higher

- Node.js and npm

- PHP 8.2 or higher

- Composer

- MySQL

- Git (Optional)

### 2️⃣ Clone the repository

```bash
git clone https://github.com/kenwrickespinosa/loan-approval-prediction.git
cd loan-approval-prediction
```

### 3️⃣ Backend setup

Navigate to Laravel folder:

```bash
cd server-api
```

Install dependencies & copy environment file:

```bash
composer install
cp .env.example .env
```

> Update `.env` with your database credentials

Generate publication key:

```bash
php artisan key:generate
```

Run database migration:

```bash
php artisan migrate
```

> Open Postman create user and provide firstname, lastname, gender=Male|Female|Other, birthdate, 
address, role=staff, email, password

Start server:

```bash
php artisan serve
```

### 4️⃣ Frontend setup

Navigate to React:

```bash
cd ../client
```

Install dependencies:

```bash
npm install
```

Start development server

```bash
npm run dev
```

## 📷 Screenshots

![Screenshot of dashboard](/assets/dashboard.png)

![Screenshot of applicants list](/assets/applicants_list.png)

![Screenshot of accepted loan](/assets/evaluation.png)

![Screenshot of not accepted loan](/assets/evaluation_not_accepted.png)

## 👤 Author

GitHub: [https://github.com/kenwrickespinosa](https://github.com/kenwrickespinosa)

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)