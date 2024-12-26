# Import Library
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pylab
import pickle
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.columns",None)
pd.set_option("display.max.rows",None)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sns.set(style="darkgrid", font_scale=1.5)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# Awal Kode
st.title("Prediksi Harga Mobil")
st.markdown("Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur yang ada.")
filePath = "../carPrice.csv"
df = pd.read_csv(filePath)

# EDA
st.header("1. Exploratory Data Analysis")
# Informasi dasar
st.subheader("1.1 Informasi Dasar Dataset")
st.write("Tampilan Awal Data:")
st.dataframe(df.head()) 
st.write("Informasi Dataset:")
st.write(f"Dimensi Data: {df.shape[0]} baris dan {df.shape[1]} kolom")
st.write("Deskripsi Statistik Data:")
st.dataframe(df.describe())
st.write("Cek Missing Value:")
st.dataframe(df.isnull().sum().to_frame().rename(columns={0: "Total Missing Values"}))
st.write("Cek Duplikasi Data:")
st.write(f"Jumlah Duplikasi Data: {df.duplicated().sum()}")
st.write("Fitur Kategorikal:")
st.dataframe(df.select_dtypes(include="object").head())
st.write("Fitur Numerikal:")
st.dataframe(df.select_dtypes(include=["int", "float"]).head())

# cleaning data
st.subheader("1.2 Pembersihan Data")
st.write("Memisahkan Nama Perusahaan Mobil:")
df["CompanyName"] = df["CarName"].apply(lambda x: x.split(" ")[0])
df.drop(columns=["CarName"], inplace=True)
st.dataframe(df.head())
st.write("Nama Perusahaan Mobil Unik:")
st.dataframe(df["CompanyName"].unique())
st.write("Perbaikan Nama Perusahaan Mobil:")
def replace(a, b): df["CompanyName"].replace(a, b, inplace=True)
replace('maxda', 'mazda')
replace('porcshce', 'porsche')
replace('toyouta', 'toyota')
replace('Nissan', 'nissan')
replace('vokswagen', 'volkswagen')
replace('vw', 'volkswagen')
st.write("Nama Perusahaan Mobil setelah Perbaikan:")
st.dataframe(df["CompanyName"].unique())

# Visualisasi
st.subheader("1.3 Visualisasi Data")
# korelasi fitur
st.write("Korelasi antar fitur")
numerical_features = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_features.corr()
fig, ax = plt.subplots(figsize=(20, 20))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# plotting data
def categorical_visualization(column, insights):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    sns.countplot(x=column, data=df, palette="Set2", order=df[column].value_counts().index, ax=axes[0])
    axes[0].set_title(f"{column} Distribusi", pad=10, fontweight="bold", fontsize=15)
    axes[0].tick_params(axis='x', rotation=90)

    sns.boxplot(x=column, y="price", data=df, palette="Set2", ax=axes[1])
    axes[1].set_title(f"{column} vs Harga", pad=10, fontweight="bold", fontsize=15)
    axes[1].tick_params(axis='x', rotation=90)

    mean_price = df.groupby(column)["price"].mean().sort_values(ascending=False)
    sns.barplot(x=mean_price.index, y=mean_price.values, palette="Set2", ax=axes[2])
    axes[2].set_title(f"Rata-rata Harga berdasarkan {column}", pad=10, fontweight="bold", fontsize=15)
    axes[2].tick_params(axis='x', rotation=90)

    st.pyplot(fig)
    st.markdown("### Insights")
    st.markdown(insights)

# Sidebar navigation
st.sidebar.title("1.3 Visualisasi Data")
options = st.sidebar.selectbox(
    "Pilih Visualisasi",
    [
        "Distribusi Harga Mobil",
        "Jumlah Mobil Berdasarkan Perusahaan",
        "Perusahaan Mobil Berdasarkan Harga",
        "Jenis Bahan Bakar",
        "Aspiration",
        "Jumlah Pintu",
        "Tipe Body Mobil",
        "Roda Penggerak",
        "Lokasi Mesin",
        "Tipe Mesin",
        "Jumlah Silinder",
        "Sistem Bahan Bakar",
        "Symboling"
    ],
)
# Logika visualisasi berdasarkan pilihan
if options == "Distribusi Harga Mobil":
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df["price"], kde=True, color="blue", ax=ax[0])
    ax[0].set_title("Distribusi Harga Mobil", fontweight="bold", fontsize=15)
    sns.boxplot(y=df["price"], palette="Set2", ax=ax[1])
    ax[1].set_title("Sebaran Harga Mobil", fontweight="bold", fontsize=15)
    st.pyplot(fig)
    st.markdown("### Insights")
    st.markdown("""
    - Harga mobil menunjukkan distribusi skewed ke kanan.
    - Ada beberapa outlier pada harga yang sangat tinggi.
    - Sebagian besar mobil memiliki harga di bawah rata-rata dataset.
    """)
elif options == "Jumlah Mobil Berdasarkan Perusahaan":
    fig, ax = plt.subplots(figsize=(14, 6))
    counts = df["CompanyName"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", ax=ax)
    ax.set_title("Jumlah Mobil Berdasarkan Perusahaan", fontweight="bold", fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)
    st.markdown("### Insights")
    st.markdown("""
    - Toyota adalah merek mobil yang paling banyak diproduksi dalam dataset.
    - Beberapa perusahaan, seperti Renault dan Mercury, memiliki data yang sangat sedikit.
    """)
elif options == "Perusahaan Mobil Berdasarkan Harga":
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(x="CompanyName", y="price", data=df, ax=ax[0], palette="Set2")
    ax[0].set_title("Perusahaan Mobil vs Harga", fontweight="bold", fontsize=15)
    ax[0].tick_params(axis='x', rotation=90)

    avg_price = df.groupby("CompanyName")["price"].mean().sort_values(ascending=False)
    sns.barplot(x=avg_price.index, y=avg_price.values, palette="Set2", ax=ax[1])
    ax[1].set_title("Harga Rata-rata Berdasarkan Perusahaan", fontweight="bold", fontsize=15)
    ax[1].tick_params(axis='x', rotation=90)
    st.pyplot(fig)
    st.markdown("### Insights")
    st.markdown("""
    - Jaguar memiliki harga rata-rata tertinggi di antara semua perusahaan.
    - Perusahaan seperti Nissan dan Renault memiliki data yang terbatas, sehingga sulit untuk dianalisis lebih lanjut.
    """)
elif options == "Symboling":
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    sns.countplot(x="symboling", data=df, palette="Set2", order=df["symboling"].value_counts().index, ax=axes[0])
    axes[0].set_title("Distribusi Symboling", pad=10, fontweight="bold", fontsize=15)
    sns.boxplot(x="symboling", y="price", data=df, palette="Set2", ax=axes[1])
    axes[1].set_title("Symboling vs Harga", pad=10, fontweight="bold", fontsize=15)
    avg_price_symboling = df.groupby("symboling")["price"].mean().sort_values(ascending=False)
    sns.barplot(x=avg_price_symboling.index, y=avg_price_symboling.values, palette="Set2", ax=axes[2])
    axes[2].set_title("Rata-rata Harga berdasarkan Symboling", pad=10, fontweight="bold", fontsize=15)
    st.pyplot(fig)
    st.markdown("### Insights")
    st.markdown("""
    - Symboling menunjukkan kategori risiko asuransi. Semakin tinggi nilai symboling, semakin tinggi risiko.
    - Harga rata-rata mobil cenderung lebih tinggi pada mobil dengan symboling rendah, menunjukkan mobil dengan risiko rendah lebih mahal.
    - Distribusi symboling merata, tetapi kategori risiko menengah lebih umum dalam dataset.
    """)
elif options == "Jenis Bahan Bakar":
    categorical_visualization(
        "fueltype",
        """
        - Mobil dengan bahan bakar gas mendominasi dataset.
        - Mobil bahan bakar diesel cenderung memiliki harga lebih tinggi dibandingkan dengan mobil gas.
        - Bahan bakar gas populer karena efisiensi biaya.
        """
    )
elif options == "Aspiration":
    categorical_visualization(
        "aspiration",
        """
        - Mobil dengan aspiration 'turbo' cenderung memiliki harga lebih tinggi dibandingkan dengan 'std'.
        - Aspiration 'turbo' lebih sedikit digunakan, kemungkinan karena biaya produksinya yang lebih tinggi.
        """
    )
elif options == "Jumlah Pintu":
    categorical_visualization(
        "doornumber",
        """
        - Mobil dengan 4 pintu paling banyak ditemukan dalam dataset.
        - Mobil dengan 2 pintu cenderung memiliki harga lebih tinggi karena seringkali didesain sporty.
        """
    )
elif options == "Tipe Body Mobil":
    categorical_visualization(
        "carbody",
        """
        - Sedan mendominasi tipe body mobil dalam dataset.
        - Mobil tipe convertible memiliki harga rata-rata tertinggi.
        - Harga mobil SUV dan wagon cukup merata dalam distribusi harga.
        """
    )
elif options == "Roda Penggerak":
    categorical_visualization(
        "drivewheel",
        """
        - Mobil berpenggerak roda depan (FWD) paling umum ditemukan.
        - Mobil berpenggerak roda belakang (RWD) cenderung lebih mahal, biasanya digunakan pada mobil sport atau mewah.
        """
    )
elif options == "Lokasi Mesin":
    categorical_visualization(
        "enginelocation",
        """
        - Lokasi mesin depan mendominasi dataset.
        - Mobil dengan mesin di belakang cenderung lebih mahal, biasanya digunakan untuk desain performa tinggi.
        """
    )
elif options == "Tipe Mesin":
    categorical_visualization(
        "enginetype",
        """
        - Tipe mesin 'ohc' paling umum digunakan.
        - Mesin 'rotor' memiliki harga rata-rata tertinggi, biasanya ditemukan pada mobil mewah atau performa tinggi.
        """
    )
elif options == "Jumlah Silinder":
    categorical_visualization(
        "cylindernumber",
        """
        - Mesin dengan 4 silinder paling umum, menunjukkan efisiensi bahan bakar yang lebih disukai.
        - Mobil dengan 8 silinder cenderung memiliki harga jauh lebih tinggi, menunjukkan hubungan dengan performa tinggi.
        """
    )
elif options == "Sistem Bahan Bakar":
    categorical_visualization(
        "fuelsystem",
        """
        - Sistem bahan bakar 'mpfi' adalah yang paling umum ditemukan.
        - Sistem 'idi' dan 'spdi' memiliki harga rata-rata yang lebih tinggi.
        """
    )
# Info tambahan
st.sidebar.info("Gunakan sidebar untuk memilih visualisasi data.")

# Preposesing
st.header("2. Preprocessing")
st.subheader("2.1 Binning")
z = round(df.groupby(["CompanyName"])["price"].agg(["mean"]), 2).T
df = df.merge(z.T, how="left", on="CompanyName")
st.write("Harga rata-rata ditambahkan ke dataset:")
st.write(df[["CompanyName", "mean"]].drop_duplicates().head())
st.subheader("2.2 Membuat Kategori Berdasarkan Harga Rata-rata")
bins = [0, 10000, 20000, 40000]
cars_bin = ['Budget', 'Medium', 'Highend']
df['CarsRange'] = pd.cut(df['mean'], bins, right=False, labels=cars_bin)
st.write("Kolom 'CarsRange' berdasarkan kategori harga:")
st.write(df[["CompanyName", "mean", "CarsRange"]].drop_duplicates().head())

# Fitur Selection
st.subheader("2.3 Memilih Fitur Penting")
selected_features = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 
                     'enginetype', 'cylindernumber', 'fuelsystem', 'wheelbase', 'carlength',
                     'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower',
                     'citympg', 'highwaympg', 'price', 'CarsRange']
new_df = df[selected_features]
st.write("Dataset dengan fitur terpilih:")
st.write(new_df.head())

# Fitur Encoding
st.subheader("2.4 Membuat Variabel Dummies")
new_df = pd.get_dummies(columns=["fueltype", "aspiration", "doornumber", "carbody", "drivewheel","enginetype", "cylindernumber", "fuelsystem", "CarsRange"], data=new_df)
st.write("Dataset setelah encoding kategorikal:")
st.write(new_df.head())

# Normalisasi
st.subheader("2.5 Scaling Fitur Numerik")
scaler = StandardScaler()
num_cols = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']
new_df[num_cols] = scaler.fit_transform(new_df[num_cols])
st.write("Dataset setelah scaling:")
st.write(new_df.head())

# split and test
st.subheader("2.6 Split Data untuk Train dan Test")
x = new_df.drop(columns=["price"])
y = new_df["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
st.write(f"Dimensi x_train: {x_train.shape}")
st.write(f"Dimensi x_test: {x_test.shape}")
st.write(f"Dimensi y_train: {y_train.shape}")
st.write(f"Dimensi y_test: {y_test.shape}")

# modelling
st.header("3. Modeling")
def model_prediction(model):
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    # R2 Score
    train_r2 = r2_score(y_train, x_train_pred) * 100
    test_r2 = r2_score(y_test, x_test_pred) * 100
    # Metrics
    mse_train = mean_squared_error(y_train, x_train_pred)
    rmse_train = mse_train ** 0.5
    mae_train = mean_absolute_error(y_train, x_train_pred)
    mape_train = (abs((y_train - x_train_pred) / y_train).mean()) * 100
    mse_test = mean_squared_error(y_test, x_test_pred)
    rmse_test = mse_test ** 0.5
    mae_test = mean_absolute_error(y_test, x_test_pred)
    mape_test = (abs((y_test - x_test_pred) / y_test).mean()) * 100
    return train_r2, test_r2, mse_train, rmse_train, mae_train, mape_train, mse_test, rmse_test, mae_test, mape_test
# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}
training_score = []
testing_score = []
results = []

for model_name, model in models.items():
    train_r2, test_r2, mse_train, rmse_train, mae_train, mape_train, mse_test, rmse_test, mae_test, mape_test = model_prediction(model)
    training_score.append(train_r2)
    testing_score.append(test_r2)
    results.append({
        "Model": model_name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "MSE Train": mse_train,
        "RMSE Train": rmse_train,
        "MAE Train": mae_train,
        "MAPE Train": mape_train,
        "MSE Test": mse_test,
        "RMSE Test": rmse_test,
        "MAE Test": mae_test,
        "MAPE Test": mape_test
    })

st.write("3.1 Hasil Model:")
results_df = pd.DataFrame(results)
st.write(results_df)
st.write("3.2 Performa Model:")
plot_df = pd.DataFrame({"Algorithms": list(models.keys()), "Training Score": training_score, "Testing Score": testing_score})
fig, ax = plt.subplots(figsize=(16, 6))
plot_df.plot(x="Algorithms", y=["Training Score", "Testing Score"], kind="bar", colormap="Set1", ax=ax)
plt.title("Performa beberapa model")
st.pyplot(fig)

# uji coba manual
st.header("4. Uji Coba")
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)
# Fitur inputan pengguna
categorical_features = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginetype','cylindernumber', 'fuelsystem', 'CarsRange']
numerical_features = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']

# Input untuk fitur kategorikal
user_input = {}
for feature in categorical_features:
    unique_values = {
        'fueltype': ['gas', 'diesel'],
        'aspiration': ['std', 'turbo'],
        'doornumber': ['two', 'four'],
        'carbody': ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'],
        'drivewheel': ['rwd', 'fwd', '4wd'],
        'enginetype': ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'],
        'cylindernumber': ['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'],
        'fuelsystem': ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi'],
        'CarsRange' : ['Medium', 'Highend', 'Budget']
    }
    user_input[feature] = st.selectbox(f"Pilih {feature}", unique_values[feature])

# Input untuk fitur numerikal
for feature in numerical_features:
    min_val, max_val = {
        'wheelbase': (86.6, 120.9),
        'carlength': (141.1, 208.1),
        'carwidth': (60.3, 72.3),
        'curbweight': (1488, 4066),
        'enginesize': (61, 326),
        'boreratio': (2.54, 3.94),
        'horsepower': (48, 288),
        'citympg': (13, 49),
        'highwaympg': (16, 54)
    }[feature]
    
    if isinstance(min_val, float) or isinstance(max_val, float):
        step = 0.1  
        default_value = round((min_val + max_val) / 2, 1)
    else:
        step = 1  
        default_value = (min_val + max_val) // 2 

    user_input[feature] = st.slider(
        f"Masukkan nilai untuk {feature}",
        min_value=min_val,
        max_value=max_val,
        value=default_value,  
        step=step
    )


all_input_df = pd.DataFrame([user_input])
all_input_df = pd.get_dummies(columns=["fueltype","aspiration","doornumber","carbody","drivewheel","enginetype","cylindernumber","fuelsystem","CarsRange"],data=all_input_df)
scaler = StandardScaler()
num_cols = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg']
all_input_df[num_cols] = scaler.fit_transform(all_input_df[num_cols])
kolom = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg', 'fueltype_diesel', 'fueltype_gas', 'aspiration_std', 'aspiration_turbo', 'doornumber_four', 'doornumber_two','carbody_convertible', 'carbody_hardtop', 'carbody_hatchback','carbody_sedan', 'carbody_wagon', 'drivewheel_4wd','drivewheel_fwd', 'drivewheel_rwd', 'enginetype_dohc','enginetype_dohcv', 'enginetype_l', 'enginetype_ohc','enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor','cylindernumber_eight', 'cylindernumber_five','cylindernumber_four', 'cylindernumber_six','cylindernumber_three', 'cylindernumber_twelve','cylindernumber_two', 'fuelsystem_1bbl', 'fuelsystem_2bbl','fuelsystem_4bbl', 'fuelsystem_idi', 'fuelsystem_mfi','fuelsystem_mpfi', 'fuelsystem_spdi', 'fuelsystem_spfi','CarsRange_Budget', 'CarsRange_Medium', 'CarsRange_Highend']
all_fitur = pd.DataFrame(columns=kolom, data=all_input_df)

# Prediksi model 
if st.button("Prediksi Harga Mobil"):
    # Prediksi harga
    predicted_price = model.predict(all_fitur)
    st.write(f"Harga mobil yang diprediksi adalah: ${predicted_price[0]:,.2f}")