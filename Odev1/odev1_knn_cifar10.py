import os
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 70)
print("CIFAR-10 k-NN SINIFLANDIRMA UYGULAMASI")
print("=" * 70)

# --------------------------------------------------
# 1) DATASET YOLU
# --------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
dataset_dir = os.path.join(base_dir, "data", "cifar-10-batches-py")

print(f"\nBeklenen veri klasörü:\n{dataset_dir}")

gerekli_dosyalar = [
    "batches.meta",
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch"
]

eksik_dosyalar = []
for dosya in gerekli_dosyalar:
    if not os.path.exists(os.path.join(dataset_dir, dosya)):
        eksik_dosyalar.append(dosya)

if len(eksik_dosyalar) > 0:
    print("\nHATA: CIFAR-10 veri seti klasörü eksik veya bazı dosyalar yok.")
    print("Eksik dosyalar:")
    for dosya in eksik_dosyalar:
        print("-", dosya)
    print("\nLütfen 'cifar-10-batches-py' klasörünü aşağıdaki dizine yerleştir:")
    print(dataset_dir)
    raise SystemExit

# --------------------------------------------------
# 2) META DOSYASI VE SINIF ADLARI
# --------------------------------------------------
print("\nVeri seti bulundu. Dosyalar okunuyor...")

with open(os.path.join(dataset_dir, "batches.meta"), "rb") as f:
    meta = pickle.load(f, encoding="bytes")

sinif_isimleri = [etiket.decode("utf-8") for etiket in meta[b"label_names"]]

print("\nSınıflar:")
for i in range(len(sinif_isimleri)):
    print(f"{i} -> {sinif_isimleri[i]}")

# --------------------------------------------------
# 3) EĞİTİM VERİSİNİ YÜKLE
# --------------------------------------------------
X_train_listesi = []
y_train_listesi = []

for i in range(1, 6):
    batch_path = os.path.join(dataset_dir, f"data_batch_{i}")
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X_train_listesi.append(batch[b"data"])
    y_train_listesi.extend(batch[b"labels"])

X_train = np.concatenate(X_train_listesi, axis=0).astype(np.int16)
y_train = np.array(y_train_listesi)

# --------------------------------------------------
# 4) TEST VERİSİNİ YÜKLE
# --------------------------------------------------
with open(os.path.join(dataset_dir, "test_batch"), "rb") as f:
    test_batch = pickle.load(f, encoding="bytes")

X_test = test_batch[b"data"].astype(np.int16)
y_test = np.array(test_batch[b"labels"])

print("\nVeri yükleme tamamlandı.")
print("Eğitim veri boyutu:", X_train.shape)
print("Test veri boyutu   :", X_test.shape)

# --------------------------------------------------
# 5) KULLANICIDAN MESAFE TÜRÜ AL
# --------------------------------------------------
print("\nMesafe türünü seçiniz:")
print("1 -> L1 (Manhattan)")
print("2 -> L2 (Öklid)")

metric = ""
while metric not in ["1", "2", "L1", "L2", "l1", "l2"]:
    metric = input("Seçiminiz: ").strip()

if metric in ["1", "L1", "l1"]:
    secilen_mesafe = "L1"
else:
    secilen_mesafe = "L2"

# --------------------------------------------------
# 6) KULLANICIDAN k DEĞERİ AL
# --------------------------------------------------
k = 0
while k <= 0 or k > len(X_train):
    try:
        k = int(input(f"k değerini giriniz (1 - {len(X_train)}): ").strip())
    except:
        k = 0

# --------------------------------------------------
# 7) HANGİ TEST GÖRÜNTÜSÜ SINIFLANDIRILACAK?
# --------------------------------------------------
test_index = -1
while test_index < 0 or test_index >= len(X_test):
    try:
        test_index = int(input(f"Sınıflandırılacak test görüntüsü indexini giriniz (0 - {len(X_test)-1}): ").strip())
    except:
        test_index = -1

print("\nSeçilen mesafe:", secilen_mesafe)
print("Seçilen k değeri:", k)
print("Seçilen test görüntüsü indexi:", test_index)

# --------------------------------------------------
# 8) TEST GÖRÜNTÜSÜNÜ AL
# --------------------------------------------------
x = X_test[test_index]
gercek_etiket = y_test[test_index]

# --------------------------------------------------
# 9) TÜM EĞİTİM VERİLERİNE OLAN UZAKLIKLARI HESAPLA
#    RAM patlamasın diye parça parça hesaplanıyor
# --------------------------------------------------
print("\nUzaklıklar hesaplanıyor...")

chunk_size = 1000
uzakliklar_parca_parca = []

for baslangic in range(0, len(X_train), chunk_size):
    bitis = baslangic + chunk_size
    egitim_parcasi = X_train[baslangic:bitis]

    if secilen_mesafe == "L1":
        parcadaki_uzakliklar = np.sum(np.abs(egitim_parcasi - x), axis=1)
    else:
        fark = egitim_parcasi.astype(np.float32) - x.astype(np.float32)
        parcadaki_uzakliklar = np.sqrt(np.sum(fark * fark, axis=1))

    uzakliklar_parca_parca.append(parcadaki_uzakliklar)

uzakliklar = np.concatenate(uzakliklar_parca_parca, axis=0)

# --------------------------------------------------
# 10) EN YAKIN k KOMŞUYU BUL
# --------------------------------------------------
en_yakin_indeksler = np.argsort(uzakliklar)[:k]
en_yakin_etiketler = y_train[en_yakin_indeksler]
en_yakin_uzakliklar = uzakliklar[en_yakin_indeksler]

# --------------------------------------------------
# 11) OYLAMA İLE SINIFI BELİRLE
# --------------------------------------------------
oy_sayilari = np.bincount(en_yakin_etiketler, minlength=len(sinif_isimleri))
tahmin_etiketi = np.argmax(oy_sayilari)

# --------------------------------------------------
# 12) SONUÇLARI YAZDIR
# --------------------------------------------------
print("\n" + "=" * 70)
print("SINIFLANDIRMA SONUCU")
print("=" * 70)
print("Gerçek sınıf :", sinif_isimleri[gercek_etiket])
print("Tahmin sınıf :", sinif_isimleri[tahmin_etiketi])

if tahmin_etiketi == gercek_etiket:
    print("Durum        : DOĞRU TAHMİN")
else:
    print("Durum        : YANLIŞ TAHMİN")

yazdirilacak_komsu_sayisi = min(20, k)

print(f"\nEn yakın {k} komşu ile işlem yapıldı.")
print(f"Ekrana sadece ilk {yazdirilacak_komsu_sayisi} komşu yazdırılıyor:\n")

for sira in range(yazdirilacak_komsu_sayisi):
    komsu_index = en_yakin_indeksler[sira]
    komsu_etiket = en_yakin_etiketler[sira]
    komsu_uzaklik = en_yakin_uzakliklar[sira]

    print(
        f"{sira + 1}. komşu -> Eğitim index: {komsu_index}, "
        f"Sınıf: {sinif_isimleri[komsu_etiket]}, "
        f"Uzaklık: {komsu_uzaklik:.4f}"
    )

print("\nOy sayıları:")
for i in range(len(sinif_isimleri)):
    print(f"{sinif_isimleri[i]}: {oy_sayilari[i]}")

# --------------------------------------------------
# 13) TEST GÖRÜNTÜSÜNÜ GÖSTER
# --------------------------------------------------
goruntu = X_test[test_index].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)

plt.figure(figsize=(4, 4))
plt.imshow(goruntu)
plt.title(
    f"Test index: {test_index}\n"
    f"Gerçek: {sinif_isimleri[gercek_etiket]} | Tahmin: {sinif_isimleri[tahmin_etiketi]}"
)
plt.axis("off")
plt.tight_layout()

cikti_yolu = os.path.join(base_dir, "sonuc_goruntu.png")
plt.savefig(cikti_yolu, dpi=150, bbox_inches="tight")
print(f"\nGörüntü başarıyla kaydedildi: {cikti_yolu}")