# Car Evaluation ML API

Bu proje, araçları özelliklerini (fiyat, kapı sayısı, güvenlik vb.) kalite sınıflandırmasına sokan uçtan uça bir ML tahmin API'sidir. Model, **Random Forest** algoritması ile eğitilmiş olup, sunucu tarafında **FastAPI** ve veri doğrulama işlemleri için **Pydantic** kullanılmıştır.

## Mimari ve Özellikler

- `scikit-learn` ile model eğitimi ve `joblib` kullanılarak ağırlıkların dışa aktarımı (.pkl)
- `FastAPI` ile yüksek performanslı REST API kurulumu
- `Pydantic` ile POST isteklerinde katı veri tipi doğrulaması (Type Validation)
- Yerleşik Swagger UI üzerinden kolay arayüz testi

## Kurulum
Projeyi bilgisayarınıza klonlayın ve bağımlılıkları yükleyin:

```bash
git clone [https://github.com/Ahmetbalkz250/randomforest_api_app.git](https://github.com/Ahmetbalkz250/randomforest_api_app.git)
cd randomforest_api_app
pip install -r requirements.txt
```

## Kullanım

Yerel sunucuyu bilgisayarınızda çalıştırmak için aşağdaki komutu kullanın:

```bash
python api.py
```

Sunucu çalıştıktan sonra tarayıcınızdan http://localhost:8000/docs adresine giderek Swagger UI üzerinden API'yi doğrudan test edebilirsiniz.

## Örnek istek (JSON Formatı)

```bash
{
  "buying_price": 2,
  "maintenance_cost": 3,
  "number_of_doors": 2,
  "number_of_persons": 1,
  "lug_boot": 1,
  "safety": 2
}
```
