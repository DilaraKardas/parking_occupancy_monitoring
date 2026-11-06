# Parking Occupancy – SAHI & YOLO26s

Bu proje, otopark videoları üzerinde **Computer Vision (Bilgisayarlı Görü)** tekniklerini kullanarak araç tespiti gerçekleştirmek ve model performansını ölçmek amacıyla geliştirilmiştir.

Projenin temelinde YOLO26s nesne tespit modeli ve SAHI (Slicing Aided Hyper Inference) kütüphanesi bulunmaktadır. SAHI sayesinde video kareleri daha küçük görüntü dilimlerine ayrılarak özellikle görüntüde küçük veya uzakta bulunan araçların tespit edilmesi hedeflenmektedir.

Model, ONNX formatındaki ağırlıkları kullanarak yerel ortamda çalıştırılmaktadır.

## 1. Projenin Amacı

Otopark görüntülerinde bulunan araçları otomatik olarak tespit etmek ve tespit sonuçlarını video üzerinde görselleştirmek.

Projenin mevcut kapsamı:

- YOLO26s ile araç tespiti.
- ONNX formatındaki model ağırlıklarını kullanarak yerel ortamda inference gerçekleştirme.
- SAHI ile görüntü dilimleme ve dilimler üzerinde nesne tespiti.
- Tespit edilen araçların bounding box ve sınıf etiketleriyle gösterilmesi.
- Video üzerinde anlık FPS değerinin görüntülenmesi.
- Modelin inference performansının benchmark edilmesi.
- Deney ayarlarının merkezi bir `config.yaml` dosyasından yönetilmesi.

## 2. Kullanılan Teknolojiler

| Teknoloji | Kullanım amacı |
|---|---|
| Python | Ana programlama dili |
| YOLO26s | Nesne tespit modeli |
| ONNX | Modelin taşınabilir inference formatı |
| SAHI | Görüntü dilimleme ve sliced inference |
| OpenCV | Video okuma, görüntü işleme ve çıktı oluşturma |
| PyTorch | Donanım kontrolü ve GPU senkronizasyonu |
| PyYAML | YAML yapılandırma dosyasını okuma |

## 3. Model Seçimi

Proje kapsamında YOLO serisinin farklı varyantları (`yolo26n`, `yolo26s` ve `yolo26m`) karşılaştırılmıştır. Tüm model testleri adil bir değerlendirme yapabilmek amacıyla **aynı cihaz (`cuda:0`)** ve **aynı ısınma karesi (`warmup_frames: 10`)** parametreleri ile gerçekleştirilmiştir. 

Yapılan değerlendirmeler sonucunda; FPS performansı, çıkarım gecikmesi (latency) ve yanlış pozitif/negatif (FP/FN) analizleri birlikte ele alındığında en dengeli ve başarılı sonucun **`yolo26s.pt`** modelinde elde edildiği görülmüş ve ana model olarak bu model seçilmiştir.

### Model Karşılaştırma Tablosu

| Model | average_latency_ms | average_fps |
| :--- | :---: | :---: |
| `yolo26n.pt` | 135.016 | 7.407 |
| `yolo26s.pt` | 137.681 | 7.263 |
| `yolo26m.pt` | 145.686 | 6.864 |

## 4. Proje Dosya Yapısı

```text
parking_occupancy_monitoring/
│
├── core/
│   ├── __pycache__/
│   ├── __init__.py
│   └── settings.py
│
├── models/
│   ├── yolo26n.pt
│   ├── yolo26s.onnx
│   └── yolo26s.pt
│
├── notebooks/
│   └── parking_lot_test.ipynb
│
├── results/
│   └── benchmarks/
│       ├── benchmark_frames.csv
│       └── benchmark_summary.csv
│
├── roi/
│   └── parking_roi.json
│
├── runs/
│   ├── detect/
│   └── parking_lot_boxes_*.mp4
│
├── scripts/
│   ├── demo.py
│   ├── export_onnx.py
│   └── resize_video.py
│
├── services/
│   ├── __pycache__/
│   ├── __init__.py
│   └── detector.py
│
├── tests/
│
├── utils/
│   ├── __pycache__/
│   ├── __init__.py
│   └── video_utils.py
│
├── videos/
│
├── .gitignore
├── config.yaml
├── main.py
├── README.md
└── requirements.txt
```

### Dosyaların görevleri

**`core/settings.py`**

Projenin merkezi yapılandırma modülüdür. `config.yaml` dosyasını okuyarak model, inference, SAHI ve benchmark parametrelerini yükler. Ayrıca çalışma ortamını kontrol ederek çıktı dizinini belirler.

**`services/detector.py`**

Modelin SAHI üzerinden yüklenmesini ve görüntü kareleri üzerinde sliced inference gerçekleştirilmesini sağlar.

İki temel fonksiyon içerir:

- `load_model()`: YOLO26s ONNX modelini SAHI aracılığıyla yükler.
- `run_inference()`: Görüntü üzerinde SAHI tabanlı nesne tespiti gerçekleştirir.

**`utils/video_utils.py`**

Video işleme sırasında kullanılan yardımcı fonksiyonları içerir:

- GPU işlemlerinin senkronizasyonu.
- FPS bilgisinin video üzerine çizilmesi.
- Video özelliklerinin okunması.
- Benchmark sonuçlarının CSV formatında kaydedilmesi.

**`main.py`**

Uygulamanın ana giriş noktasıdır. Video okuma, model yükleme, inference, araç tespiti, bounding box çizimi, FPS hesaplama ve sonuçların kaydedilmesi işlemlerini yönetir.

**`config.yaml`**

Model ve deney parametrelerinin merkezi olarak tanımlandığı yapılandırma dosyasıdır.

**`export_onnx.py`**

Modelin ONNX formatına dönüştürülmesi için kullanılan yardımcı betiktir.

**`resize_video.py`**

Giriş videosunun yeniden boyutlandırılması için kullanılan yardımcı betiktir.

## 5. Araç Sınıfları

Proje, COCO veri kümesinde tanımlanan aşağıdaki dört araç sınıfına odaklanmaktadır:

| Sınıf | COCO Class ID |
|---|---:|
| Car | 2 |
| Motorcycle | 3 |
| Bus | 5 |
| Truck | 7 |

Sınıf filtrelemesi `core/settings.py` içerisinde tanımlanan `VEHICLE_CLASSES` kümesiyle gerçekleştirilir.

```python
VEHICLE_CLASSES = {2, 3, 5, 7}
```

Bu filtre, inference sonucunda dönen tahminler arasından yalnızca ilgili araç sınıflarının işlenmesini sağlar.

## 6. SAHI Yapılandırması

SAHI, görüntüyü daha küçük dilimlere ayırarak her dilim üzerinde ayrı nesne tespiti gerçekleştirmeyi sağlar.

Bu yaklaşım, özellikle yüksek çözünürlüklü görüntülerde küçük nesnelerin tespit edilmesine yardımcı olabilir.

Projede kullanılan temel SAHI parametreleri:

| Parametre | Değer |
|---|---:|
| Slice Height | 640 |
| Slice Width | 640 |
| Overlap Height Ratio | 0.2 |
| Overlap Width Ratio | 0.2 |
| Postprocess Type | GREEDYNMM |
| Match Metric | IOS |
| Match Threshold | 0.5 |

**GREEDYNMM**, farklı görüntü dilimlerinde bulunan örtüşen tahminlerin birleştirilmesi için kullanılan post-processing yaklaşımıdır.

**IOS (Intersection over Smaller Area)**, iki bounding box arasındaki örtüşmeyi küçük olan kutunun alanına göre değerlendiren eşleştirme metriğidir.

## 7. Model ve Inference Parametreleri

Model ve inference ayarları `config.yaml` üzerinden yönetilir.

Mevcut yapılandırma:

```yaml
model:
  path: "models/yolo26s.onnx"
  device: "cuda:0"

inference:
  conf: 0.5
  iou: 0.5
  imgsz: 640

sahi:
  slice_height: 640
  slice_width: 640
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2
  postprocess_type: "GREEDYNMM"
  postprocess_match_metric: "IOS"
  postprocess_match_threshold: 0.5

benchmark:
  warmup_frames: 10
  measured_frames: 100
```

### Inference parametreleri

- **Confidence Threshold:** Bir tespitin kabul edilmesi için gereken minimum güven skoru.
- **IoU Threshold:** Modelin Non-Maximum Suppression (NMS) aşamasında kullandığı örtüşme eşiği.
- **Image Size:** Model inference sırasında kullanılan görüntü boyutu.

Bu parametreler değiştirilerek farklı deneyler gerçekleştirilebilir.

## 8. Benchmark ve Performans Ölçümü

Model performansını daha tutarlı ölçebilmek için benchmark sürecinde iki aşama uygulanmaktadır.

1. **Warm-up:** İlk 10 video karesi ısınma amacıyla işlenir ve performans ölçümüne dahil edilmez.
2. **Measurement:** Sonraki 100 kare üzerinde performans ölçümü gerçekleştirilir.

GPU üzerinde doğru zaman ölçümü elde edebilmek için inference öncesinde ve sonrasında GPU senkronizasyonu uygulanır.

### Ölçülen metrikler

**Average Latency (ms/frame)**

Bir video karesinin işlenmesi için geçen ortalama süreyi ifade eder.

**Average FPS**

Saniyede işlenebilen ortalama kare sayısını ifade eder.

Hesaplamalar:

```text
Average Latency = Total Inference Time / Measured Frames

Average FPS = Measured Frames / Total Inference Time
```

### Ölçüm kapsamı

Benchmark süresine aşağıdaki işlemler dahildir:

- BGR → RGB renk uzayı dönüşümü.
- SAHI sliced inference işlemi.

Aşağıdaki işlemler benchmark süresine dahil değildir:

- Bounding box çizimi.
- FPS metninin görüntü üzerine yazılması.
- Video çıktısının diske kaydedilmesi.

Bu ayrım, model inference performansını video yazma ve görselleştirme maliyetlerinden bağımsız değerlendirmeyi amaçlamaktadır.

Benchmark sonuçları CSV formatında kaydedilir:

- `benchmark_summary_*.csv`: Genel performans özeti.
- `benchmark_frames_*.csv`: Kare bazında latency ve FPS değerleri.

## 9. Model Ağırlıkları ve Kurulum

Çıkarım (inference), video işleme ve takip hatlarını (pipelines) çalıştırabilmek için önceden eğitilmiş model ağırlıklarının `models/` dizinine ve örnek test videosunun da `videos/` dizinine eklenmesi gereklidir.

### Ağırlıkları ve Örnek Videoyu İndirin
Model ağırlıklarını ve test videolarını sıfırdan indirmekle uğraşmamak için aşağıdaki Google Drive bağlantısını kullanabilirsiniz:

* **[Google Drive - Model Ağırlıkları ve Örnek Videoları İndir](https://drive.google.com/drive/folders/1SefYxWOFl9EqymYb1tRAaftRkoSKdkJ_?usp=sharing)**

İndirdiğiniz dosyaları proje dizininde aşağıdaki konumlara yerleştirin:

Proje dizin yapınızın şu şekilde olması gerekir:
```text
parking_occ/
├── models/
│   └── yolo26s.onnx
├── videos/
│   └── parking_lot.mp4
├── core/
├── scripts/
└── README.md

### Sanal ortam oluşturma

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Bağımlılıkları yükleme

```bash
pip install -r requirements.txt
```

Not: PyTorch ve CUDA kurulumunun kullanılan donanımla uyumlu olması gerekir. ONNX modelinin Ultralytics/SAHI entegrasyonuyla çalışması için gerekli bağımlılıkların da kurulu olması gerekir.

## 10. Projeyi Çalıştırma

Proje ana dizinindeyken:

```bash
python main.py
```

Program:

1. Yapılandırma dosyasını okur.
2. Modeli yükler.
3. Giriş videosunu açar.
4. SAHI ile araç tespiti gerçekleştirir.
5. Tespitleri video üzerinde görselleştirir.
6. FPS bilgisini gösterir.
7. Benchmark sonuçlarını CSV dosyalarına kaydeder.

## 11. Çıktılar

Video işleme sonucunda:

- Araçların bounding box ve sınıf etiketleriyle gösterildiği çıktı videosu.
- Kare bazında performans ölçümlerini içeren CSV dosyası.
- Genel benchmark sonuçlarını içeren CSV dosyası.

Çıktı dosyaları `core/settings.py` içerisinde belirlenen `OUTPUT_DIR` dizinine kaydedilir.

## 12. Mevcut Geliştirme Durumu

- [x] YOLO26s modelinin kullanılması.
- [x] Modelin ONNX formatında çalıştırılması.
- [x] SAHI entegrasyonu.
- [x] Video üzerinden araç tespiti.
- [x] Araç sınıflarının filtrelenmesi.
- [x] Bounding box ve sınıf etiketlerinin görselleştirilmesi.
- [x] Anlık FPS gösterimi.
- [x] Benchmark altyapısının oluşturulması.
- [x] Deney parametrelerinin YAML üzerinden yönetilmesi.
- [x] Yerel ortamda model inference işlemi.

## 13. Gelecek Geliştirmeler

- [ ] Otopark alanlarının tanımlanması.
- [ ] Park alanlarının dolu/boş durumunun belirlenmesi.
- [ ] Araçların park alanlarıyla ilişkilendirilmesi.
- [ ] Farklı araç tespit modellerinin performans karşılaştırması.
- [ ] Ground truth anotasyonlarıyla Precision, Recall ve mAP değerlendirmesi.
- [ ] Araç takibi ve mükerrer sayımın önlenmesi.
- [ ] Gerçek zamanlı veya canlı kamera akışının desteklenmesi.

---

**Proje:** Parking Occupancy Detection  
**Model:** YOLO26s (ONNX)  
**Inference:** SAHI + Ultralytics  
**Geliştirme Dili:** Python
