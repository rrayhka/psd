# Import Library
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.columns",None)
pd.set_option("display.max.rows",None)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sns.set(style="darkgrid", font_scale=1.5)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
from streamlit_option_menu import option_menu
filePath = "carPrice.csv"
df = pd.read_csv(filePath)


def main():
    with st.sidebar:
        page = option_menu("Pilih Halaman", ["Home", "Data Understanding", "Preprocessing", "Model & Evaluasi", "Testing"], default_index=0)

    if page == "Home":
        show_home()
    elif page == "Data Understanding":
        dfUnderstanding = show_understanding()
        st.session_state["dfUnderstanding"] = dfUnderstanding
    elif page == "Preprocessing":
        if "dfUnderstanding" in st.session_state:
            # show_preprocessing(st.session_state["dfUnderstanding"])
            st.session_state["dfPreprocessing"] = show_preprocessing(st.session_state["dfUnderstanding"])
        else:
            st.warning("Silakan kunjungi halaman Data Understanding terlebih dahulu dan baca sampai selesai.")
    elif page == "Model & Evaluasi":
        if "dfPreprocessing" in st.session_state:
            st.session_state["dfModel"] = show_model(st.session_state["dfPreprocessing"])
        else:
            st.warning("Silakan kunjungi halaman Preposesing terlebih dahulu dan baca sampai selesai.")
    elif page == "Testing":
        show_testing()


def show_home():
    st.title(
        "Prediksi Harga Mobil menggunakan metode Linear Regression, Random Forest Regressor, dan XGBoost Regressor")

    # Explain what is Decision Tree
    st.header("Apa itu Prediksi?")
    st.write("Prediksi adalah suatu proses untuk memperkirakan nilai dari suatu variabel berdasarkan variabel lainnya.")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memprediksi harga mobil berdasarkan beberapa variabel hasil seleksi fitur yang penting.")

    # Explain the data
    st.header("Data")
    st.write(
        "Data yang digunakan adalah data mobil yang diambil dari Kaggle. Data ini berisi 26 variabel dan 205 dataset.")

    # Explain the process of Decision Tree
    st.header("Tahapan Proses Klasifikasi K-Nearest Neighbor")
    st.write("1. **Data Understanding atau Pemahaman Data**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Pemodelan & Evaluasi Model**")
    st.write("5. **Implementasi**")




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

def scatter_plot(cols):
    x = 1
    fig, axs = plt.subplots(1, len(cols), figsize=(15, 6))
    
    for i, col in enumerate(cols):
        sns.scatterplot(x=col, y="price", data=df, color="blue", ax=axs[i])
        axs[i].set_title(f"{col} vs Harga", fontweight="bold", fontsize=12, pad=10)
    
    plt.tight_layout()
    st.pyplot(fig)

def scatter_plot2(cols):
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 6), sharey=True)  
    for i, col in enumerate(cols):
        sns.scatterplot(ax=axes[i], x=col, y="price", data=df, color="blue")
        axes[i].set_title(f"{col} vs Price", fontweight="black", fontsize=20, pad=10)
    plt.tight_layout()
    st.pyplot(fig)

def show_understanding():
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
    st.subheader("1.3.1 Korelasi antar fitur")
    numerical_features = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_features.corr()
    fig, ax = plt.subplots(figsize=(20, 20))  
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # Distribusi harga mobil
    st.subheader("1.3.2 Distribusi Harga Mobil")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df["price"], color="red", kde=True, ax=axes[0])
    axes[0].set_title("Distribusi Harga Mobil", fontweight="bold", pad=20, fontsize=15)
    sns.boxplot(y=df["price"], palette="Set2", ax=axes[1])
    axes[1].set_title("Sebaran Harga Mobil", fontweight="bold", pad=20, fontsize=15)
    st.pyplot(fig)
    st.write(df["price"].agg(["min","mean","median","max","std","skew"]).to_frame().T)
    st.markdown(
        """
        ### Insights
        1. Terlihat dengan jelas kalau **fitur Harga Mobil** ini condong ke kanan.
        2. Jelas juga terlihat kalau ada perbedaan besar antara **nilai rata-rata (mean) dan median**.
        3. Dari sini, bisa tahu kalau **kebanyakan harga mobil di bawah 14.000**.
        4. Selain itu, **skewness harga mobil di atas 1,5**, artinya data ini **tersebar cukup luas.**
        """
    )

    st.subheader("1.3.3 Visualisasi Total Jumlah Mobil yang Terjual Berdasarkan Perusahaan yang Berbeda")
    fig, ax = plt.subplots(figsize=(15, 6))
    counts = df["CompanyName"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)

    # Menambahkan label dan judul
    ax.set_xlabel("Perusahaan Mobil")
    ax.set_ylabel("Total jumlah mobil terjual")
    ax.set_title("Total mobil diproduksi oleh perusahaan", pad=20, fontweight="bold", fontsize=20)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    st.write("Total mobil yang terjual berdasarkan 3 perusahaan terendah: Jaguar, Renault, dan Mercury")
    st.write(df[df["CompanyName"]=="jaguar"])
    st.write(df[df["CompanyName"]=="renault"])
    st.write(df[df["CompanyName"]=="mercury"])
    st.markdown(

        """
        ### Insights
        1. Perusahaan **Toyota** menjual mobil paling banyak.
        2. Jadi, bisa dibilang kalau **Toyota** itu perusahaan favorit pelanggan.
        3. **Jaguar**, **Renault**, dan **Mercury** punya data yang sangat sedikit, jadi nggak bisa ambil kesimpulan soal perusahaan dengan penjualan mobil paling sedikit.

        """
    )

    st.subheader("1.3.4 Visualisasi Perusahaan Mobil berdasarkan Harga")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # Plot Boxplot
    sns.boxplot(x="CompanyName", y="price", data=df, ax=axs[0])
    axs[0].set_title("Perusahaan Mobil vs Harga", pad=10, fontweight="bold", fontsize=15)
    axs[0].tick_params(axis='x', rotation=90)

    # Plot Barplot
    x = pd.DataFrame(df.groupby("CompanyName")["price"].mean().sort_values(ascending=False))
    sns.barplot(x=x.index, y="price", data=x, ax=axs[1])
    axs[1].set_title("Perusahaan Mobil vs Harga Rerata", pad=10, fontweight="bold", fontsize=15)
    axs[1].tick_params(axis='x', rotation=90)

    # Menampilkan plot dengan layout rapat
    plt.tight_layout()

    # Tampilkan plot di Streamlit
    st.pyplot(fig)
    st.write("Perusahaan mobil VS harga")
    st.write("Mengecek Mercury, Nissan, dan Renault")
    st.write(df[df["CompanyName"]=="mercury"])
    # st.write(df[df["CompanyName"]=="nissan"])
    st.write(df[df["CompanyName"]=="renault"])

    st.markdown(
        """
    ### Insights
    1. **Jaguar** & **Buick** kelihatannya punya mobil dengan rentang harga paling tinggi.
    2. Perusahaan mobil seperti **Renault**, & **Mercury** cuma punya 1-2 data.
    3. Jadi, nggak bisa ambil kesimpulan soal perusahaan mobil dengan rentang harga terendah.

    **Note**
    * Karena ada terlalu banyak kategori di fitur perusahaan mobil, bisa bikin fitur baru **Company Price Range** 
    yang menunjukkan rentang harga seperti **Low Range, Medium Range, High Range**.
    """
    )


    # Logika visualisasi berdasarkan pilihan
    st.subheader("1.3.5 Visualisasi Data Berdasarkan Pilihan:")
    option = st.selectbox("Pilih Fitur", df.columns)
    if df[option].dtype == "object":
        if option == "fueltype":
            categorical_visualization(
                "fueltype",
                """
                - Mobil dengan bahan bakar gas mendominasi dataset.
                - Mobil bahan bakar diesel cenderung memiliki harga lebih tinggi dibandingkan dengan mobil gas.
                - Bahan bakar gas populer karena efisiensi biaya.
                """
            )
        elif option == "aspiration":
            categorical_visualization(
                "aspiration",
                """
                - Mobil dengan aspiration 'turbo' cenderung memiliki harga lebih tinggi dibandingkan dengan 'std'.
                - Aspiration 'turbo' lebih sedikit digunakan, kemungkinan karena biaya produksinya yang lebih tinggi.
                """
            )
        elif option == "doornumber":
            categorical_visualization(
                "doornumber",
                """
                - Mobil dengan 4 pintu paling banyak ditemukan dalam dataset.
                - Mobil dengan 2 pintu cenderung memiliki harga lebih tinggi karena seringkali didesain sporty.
                """
            )
        elif option == "carbody":
            categorical_visualization(
                "carbody",
                """
                - Sedan mendominasi tipe body mobil dalam dataset.
                - Mobil tipe convertible memiliki harga rata-rata tertinggi.
                - Harga mobil SUV dan wagon cukup merata dalam distribusi harga.
                """
            )
        elif option == "drivewheel":
            categorical_visualization(
                "drivewheel",
                """
                - Mobil berpenggerak roda depan (FWD) paling umum ditemukan.
                - Mobil berpenggerak roda belakang (RWD) cenderung lebih mahal, biasanya digunakan pada mobil sport atau mewah.
                """
            )
        elif option == "enginelocation":
            categorical_visualization(
                "enginelocation",
                """
                - Lokasi mesin depan mendominasi dataset.
                - Mobil dengan mesin di belakang cenderung lebih mahal, biasanya digunakan untuk desain performa tinggi.
                """
            )
            st.write(df[df["enginelocation"]=="rear"])
            st.markdown(
                """
                ### Insights      
                1. Mayoritas mobil memiliki lokasi mesin di **depan**.

                **Insights**  
                1. Perlu dicatat bahwa hanya ada **3 data-poin untuk kategori rear**.  
                2. Oleh karena itu, tidak bisa menarik kesimpulan apapun tentang harga mobil berdasarkan lokasi mesin.  
                3. Jika diperlukan, fitur ini bisa dihapus sebelum proses pelatihan model, karena dapat menyebabkan **overfitting**.
                """
            )
        elif option == "enginetype":
            categorical_visualization(
                "enginetype",
                """
                - Tipe mesin 'ohc' paling umum digunakan.
                - Mesin 'rotor' memiliki harga rata-rata tertinggi, biasanya ditemukan pada mobil mewah atau performa tinggi.
                """
            )
            st.write("Mengecek enginetype dohcv dan rotor")
            st.write(df[df["enginetype"]=="dohcv"])
            st.write(df[df["enginetype"]=="rotor"])
            st.markdown(
                """
                ### Insights
                **Insights**  
                    1. Mobil dengan mesin **Overhead Camshaft (OHC)** paling banyak terjual.  
                    2. Hanya ada satu mobil yang terjual dengan tipe mesin **DOHCv**.  
                    3. Data untuk tipe mesin **DOHCv** dan **Rotor** sangat sedikit, jadi bisa dikatakan bahwa mobil dengan tipe mesin **OHC** cenderung lebih mahal.  
                    4. Mobil dengan mesin **Overhead Camshaft (OHC)** adalah mobil yang paling terjangkau.
                """)
        elif option == "cyclindernumber":
            categorical_visualization(
                "cylindernumber",
                """
                - Mesin dengan 4 silinder paling umum, menunjukkan efisiensi bahan bakar yang lebih disukai.
                - Mobil dengan 8 silinder cenderung memiliki harga jauh lebih tinggi, menunjukkan hubungan dengan performa tinggi.
                """
            )
            st.write("Mengecek cyclindernumber 3 dan 12")
            st.write(df[df["cylindernumber"]=="three"])
            st.write(df[df["cylindernumber"]=="twelve"])
            st.markdown(
                """
                ### Insights
                1. Mayoritas mobil memiliki **empat** silinder, diikuti oleh mobil dengan **enam** silinder.  
                2. Hanya ada satu data-poin untuk mobil dengan **tiga** silinder dan **dua belas** silinder.  
                3. Mobil dengan **delapan** silinder adalah mobil yang paling mahal, diikuti oleh mobil dengan **enam** silinder.
                """
            )
        elif option == "fuelsystem":
            categorical_visualization(
                "fuelsystem",
                """
                - Sistem bahan bakar 'mpfi' adalah yang paling umum ditemukan.
                - Sistem 'idi' dan 'spdi' memiliki harga rata-rata yang lebih tinggi.
                """
            )
            st.write("Mengecek fuelsystem mfi dan spfi")
            st.write(df[df["fuelsystem"]=="mfi"])
            st.write(df[df["fuelsystem"]=="spfi"])
            st.markdown(
                """
                ### Insights  
                1. Mayoritas mobil menggunakan sistem bahan bakar **MPFI** dan **2BBL**.  
                2. Mobil dengan sistem bahan bakar **MPFI** adalah mobil yang paling mahal, diikuti oleh mobil dengan sistem bahan bakar **IDI**.  
                3. Hanya ada satu data-poin untuk mobil dengan sistem bahan bakar **MFI** dan **SPFI**, sehingga tidak bisa menarik kesimpulan lebih lanjut dari data ini.
                """
            )
        
    else:
        if option == "car_ID":
            st.markdown("car_ID tidak memberikan informasi yang berguna, jadi saya bisa menghapus fitur ini.")
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            sns.histplot(df[option], kde=True, ax=axes[0])
            axes[0].set_title(f"Distribusi {option}", pad=10, fontweight="bold", fontsize=15)
            sns.scatterplot(x=option, y="price", data=df, ax=axes[1])
            axes[1].set_title(f"{option} vs Harga", pad=10, fontweight="bold", fontsize=15)
            st.pyplot(fig)

        elif option == "symboling":
            st.markdown(
                """
                ### Observasi
                1. **Symboling** menggambarkan sejauh mana mobil dianggap lebih berisiko dibandingkan dengan harga yang ditawarkan.  
                2. Nilai **symboling** berkisar antara **-3 hingga +3**, dengan nilai negatif yang lebih tinggi menunjukkan risiko yang lebih tinggi dan nilai positif yang lebih tinggi menunjukkan risiko yang lebih rendah.  
                3. Dengan kata lain, mobil dengan **symboling -3** dianggap **lebih berisiko** dibandingkan dengan mobil yang memiliki symboling +3, dan kemungkinan besar memiliki **harga yang lebih rendah** akibatnya.
                """
            )
            categorical_visualization(
                "symboling", 
                """
                1. Terlihat dengan jelas bahwa mobil dengan **symboling 0** atau **1** lebih banyak dipilih oleh pembeli.  
                2. Selain itu, mobil dengan **symboling -1, 0, 3** cenderung lebih mahal.
                """
            )
        elif option in ["carlength","carwidth","carheight"]:
            scatter_plot(["carlength","carwidth","carheight"])
            st.markdown(
                """
                    ### Insights
                    1. Terlihat dengan jelas bahwa fitur **carlength** dan **carwidth** memiliki korelasi yang tinggi dengan fitur **price**.  
                    2. Jadi, bisa disimpulkan bahwa **peningkatan panjang dan lebar mobil** berbanding lurus dengan **peningkatan harga**.  
                    3. Dari grafik **carlength vs price**, tidak bisa menarik kesimpulan karena data-point-nya terlalu tersebar.  
                    4. Karena **CarHeight** tidak mempengaruhi **Harga**, saya akan  mempertimbangkan untuk menghapus fitur ini.
                """
            )
        elif option in ["enginesize","boreratio","stroke"]:
            st.markdown(
                """
                Pada bagian ini, saya akan membuat visualisasi untuk fitur EngineSize (Ukuran Mesin), BoreRatio (Rasio Diameter Silinder), dan Stroke (Langkah Pistons) untuk melihat bagaimana ketiga fitur ini berhubungan dengan harga mobil. 
                Visualisasi ini dapat membantu saya memahami apakah ukuran mesin yang lebih besar, rasio bore yang lebih tinggi, atau langkah piston yang lebih panjang berhubungan dengan harga mobil yang lebih mahal atau lebih murah.
                """
            )
            scatter_plot(["enginesize","boreratio","stroke"])
            st.markdown(
                """
                ### Insights  
                1. Terlihat dengan jelas bahwa **EngineSize** memiliki korelasi yang tinggi dengan fitur **price**. Dengan kata lain, semakin besar ukuran mesin, semakin tinggi harga mobil.  
                2. Dari grafik **BoreRatio vs Price**, meskipun korelasi antara keduanya tidak terlalu kuat, tetap ada hubungan yang jelas. Jadi, dapat disimpulkan bahwa semakin tinggi **BoreRatio**, semakin tinggi pula harga mobil.  
                3. Dari grafik **Stroke vs Price**, tidak bisa ditarik kesimpulan karena data-point-nya terlalu tersebar.  
                4. Karena **Stroke** tidak mempengaruhi **Harga** dengan signifikan, saya bisa mempertimbangkan untuk menghapus fitur ini.
                """
            )
        elif option in ["compressionratio", "horsepower", "peakrpm"]:
            st.markdown(
                """
                Pada bagian ini, saya akan membuat visualisasi untuk fitur CompressionRatio (Rasio Kompresi), Horsepower (Tenaga Kuda), dan PeakRPM (Putaran Mesin Maksimum) untuk melihat bagaimana ketiga fitur ini berhubungan dengan harga mobil. 
                Visualisasi ini akan membantu saya memahami apakah peningkatan rasio kompresi, tenaga kuda, atau putaran mesin maksimum berhubungan dengan harga mobil yang lebih tinggi atau lebih rendah.
                """
            )
            scatter_plot(["compressionratio","horsepower","peakrpm"])
            st.markdown(
                """
                ### Insights  
                1. Terlihat dengan jelas bahwa **Horsepower** memiliki korelasi yang tinggi dengan **price**. Jadi, semakin besar **Horsepower**, semakin tinggi harga mobil.  
                2. Dari visualisasi **CompressionRatio vs Price** dan **PeakRPM vs Price**, tidak bisa menarik kesimpulan karena data-point-nya terlalu tersebar.  
                3. Karena **CompressionRatio** dan **PeakRPM** tidak mempengaruhi harga secara signifikan, saya bisa mempertimbangkan untuk menghapus kedua fitur ini.
                """
            )
        elif option in ["wheelbase","curbweight"]:
            st.markdown(
                """
                Pada bagian ini, saya akan membuat visualisasi untuk fitur WheelBase (Jarak Sumbu Roda) dan CurbWeight (Berat Mobil Kosong) untuk melihat bagaimana keduanya berhubungan dengan harga mobil.
                """
            )
            scatter_plot2(["wheelbase","curbweight"])
            st.markdown(
                """
                ### Insights  
                1. **CurbWeight** memiliki korelasi yang tinggi dengan **Price**. Jadi, semakin berat mobil (CurbWeight), semakin tinggi pula harga mobil.  
                2. Dari grafik **WheelBase vs Price**, meskipun korelasinya tidak terlalu kuat, masih ada hubungan yang cukup jelas. Dengan peningkatan **WheelBase**, harga mobil cenderung juga meningkat.
                """
            )   
        elif option in ["citympg","highwaympg"]:
            st.markdown(
                """
                Pada bagian ini, saya akan membuat visualisasi untuk fitur CityMPG (Efisiensi Bahan Bakar di Kota) dan HighwayMPG (Efisiensi Bahan Bakar di Jalan Raya) untuk melihat bagaimana kedua fitur ini berhubungan dengan harga mobil.
                """
            )
            scatter_plot2(["citympg","highwaympg"])
            st.markdown(
                """
                ### Insights  
                1. Terlihat dengan jelas bahwa **CityMPG** dan **HighwayMPG** memiliki **korelasi negatif** dengan **harga**.  
                2. Dengan kata lain, semakin tinggi **CityMPG** dan **HighwayMPG**, semakin rendah harga mobil tersebut.  
                3. Oleh karena itu, kedua fitur **CityMPG** dan **HighwayMPG** cukup berguna untuk prediksi harga mobil.
                """
            )

    st.subheader("1.4 List Fitur yang akan digunakan")
    st.markdown(
        """
        **List fitur kategorikal yang berguna.**
        1. `CompanyName`
        2. `Fuel Type`
        3. `Aspiration`
        4. `Door Number`
        5. `Car Body`
        6. `Drive Wheel`
        7. `Engine Type`
        8. `Cyclinder Number`
        9. `Fuel System`

        **List fitur numerikal yang berguna.**
        1. `Wheelbase`
        2. `Carlength`
        3. `Carwidth`
        4. `Curbeweight`
        5. `Enginesize`
        6. `Boreratio`
        7. `Horsepower`
        8. `citympg`
        9. `Highwaympg`
        10. `Price`
        """
    )
    return df

def show_preprocessing(df):
    st.header("2. Preprocessing")
    st.subheader("2.1 Binning")
    st.markdown("* Seperti yang telah dibuat insight sebelumnya, hal yang tepat yakni membagi nama perusahaan mobil menjadi berbagai rentang harga, seperti **Rentang Rendah**, **Rentang Menengah**, dan **Rentang Tinggi**.")
    z = round(df.groupby(["CompanyName"])["price"].agg(["mean"]), 2).T
    st.write("Rata-rata harga mobil berdasarkan nama perusahaan:")
    st.write(z)
    st.markdown("""
                **Catatan**  
                * Pada output di atas, saya telah mengambil **harga rata-rata untuk setiap perusahaan mobil individu**.  
                * Sekarang, perlu ditambahkan nilai rata-rata ini sebagai kolom baru dalam dataset tugas ini.
                """)
    df = df.merge(z.T, how="left", on="CompanyName")
    st.write("Harga rata-rata ditambahkan ke dataset:")
    st.write(df[["CompanyName", "mean"]].drop_duplicates().head())
    st.subheader("2.2 Membuat Kategori Berdasarkan Harga Rata-rata")
    bins = [0, 10000, 20000, 40000]
    cars_bin = ['Budget', 'Medium', 'Highend']
    df['CarsRange'] = pd.cut(df['mean'], bins, right=False, labels=cars_bin)
    st.write("Kolom 'CarsRange' berdasarkan kategori harga:")
    st.write(df[["CompanyName", "mean", "CarsRange"]].drop_duplicates().head())

    st.subheader("2.3 Menghapus Fitur yang Tidak Diperlukan (seleksi fitur)")
    new_df = df[['fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem'
                ,'wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg',
                'price','CarsRange']]
    st.write("Data setelah seleksi fitur:")
    st.write(new_df.head())
    st.write("Dimensi data setelah seleksi fitur:")
    st.write(new_df.shape)

    st.subheader("2.4 Encoding")
    st.write("Mengecek unik data disetiap kolom:")
    # Pilih kolom bertipe object (kategorikal)
    categorical_cols = new_df.select_dtypes(include="object")

    # Cek nilai unik untuk setiap kolom
    for col in categorical_cols.columns:
        st.write(f"Kolom: {col}")
        st.write(new_df[col].unique())
        
    st.write("-" * 30)
    st.write("Mengecek unik data pada kolom 'CarsRange':")
    st.write(new_df["CarsRange"].unique())

    st.write("Mengubah data kategorikal menjadi numerik:")
    new_df = pd.get_dummies(columns=["fueltype","aspiration","doornumber","carbody","drivewheel","enginetype",
                                "cylindernumber","fuelsystem","CarsRange"],data=new_df)
    st.write("Dataset setelah encoding kategorikal:")
    st.write(new_df.head())

    st.subheader("2.5 Scaling Fitur Numerik")
    scaler = StandardScaler()
    num_cols = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 
    'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']
    new_df[num_cols] = scaler.fit_transform(new_df[num_cols])
    st.write("Dataset setelah scaling:")
    st.write(new_df.head())

    return new_df

def show_model(new_df):
    st.header("3. Modelling")
    st.subheader("3.1 Split Data untuk Train dan Test")
    x = new_df.drop(columns=["price"])
    y = new_df["price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    st.write(f"Dimensi x_train: {x_train.shape}")
    st.write(f"Dimensi x_test: {x_test.shape}")
    st.write(f"Dimensi y_train: {y_train.shape}")
    st.write(f"Dimensi y_test: {y_test.shape}")

    st.subheader("3.2 Training Model")
    def model_prediction(model):
        model.fit(x_train, y_train)
        x_train_pred = model.predict(x_train)
        x_test_pred = model.predict(x_test)
        # R2 Score
        train_r2 = r2_score(y_train, x_train_pred) * 100
        test_r2 = r2_score(y_test, x_test_pred) * 100
        # Metrics
        # mse_train = mean_squared_error(y_train, x_train_pred)
        # rmse_train = mse_train ** 0.5
        # mae_train = mean_absolute_error(y_train, x_train_pred)
        # mape_train = (abs((y_train - x_train_pred) / y_train).mean()) * 100
        # mse_test = mean_squared_error(y_test, x_test_pred)
        # rmse_test = mse_test ** 0.5
        # mae_test = mean_absolute_error(y_test, x_test_pred)
        # mape_test = (abs((y_test - x_test_pred) / y_test).mean()) * 100
        # return train_r2, test_r2, mse_train, rmse_train, mae_train, mape_train, mse_test, rmse_test, mae_test, mape_test
        return train_r2, test_r2
    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }
    st.write("Model yang digunakan:")
    st.write(models)
    training_score = []
    testing_score = []
    results = []
    for model_name, model in models.items():
        train_r2, test_r2 = model_prediction(model)
        training_score.append(train_r2)
        testing_score.append(test_r2)
        results.append({
            "Model": model_name,
            "Train R2": train_r2,
            "Test R2": test_r2,
            # "MSE Train": mse_train,
            # "RMSE Train": rmse_train,
            # "MAE Train": mae_train,
            # "MAPE Train": mape_train,
            # "MSE Test": mse_test,
            # "RMSE Test": rmse_test,
            # "MAE Test": mae_test,
            # "MAPE Test": mape_test
        })

    st.write("3.1 Hasil Model:")
    results_df = pd.DataFrame(results)
    st.write(results_df)

    st.header("4. Evaluasi Performa Model")
    st.write("Matriks Evaluasi menggunakan R2 Score dikarenakan saya ingin mengecek variabilitas harga mobil yang dapat dijelaskan oleh fitur-fitur yang ada.")
    plot_df = pd.DataFrame({"Algorithms": list(models.keys()), "Training Score": training_score, "Testing Score": testing_score})
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_df.plot(x="Algorithms", y=["Training Score", "Testing Score"], kind="bar", colormap="Set1", ax=ax)
    plt.title("Performa beberapa model")
    st.pyplot(fig)
    st.markdown(
        """
        - Model **Random Forest** memberikan performa tertinggi sekitar **95%** pada test berdasarkan $R^2$.
        - Performa model **XGBoost** juga sangat baik.
        - Jadi, saya dapat menggunakan kedua model ini untuk memprediksi harga mobil.
        - saya lebih prefer menggunakan random forest dikarenakan tidak overfitting, terlihat R^2 score pada training dan testing tidak terlalu jauh.   
        """
    )

def show_testing(): 
    st.header("5. Uji Coba")
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

if __name__ == "__main__":
    main()
